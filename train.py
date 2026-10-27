import argparse
import os
import random
import time

import torch
import torch.optim as optim
import numpy as np
from src.data.dataloader import create_dataloader
from src.models.dimmrec import IntegratedMultimodalModel, compute_losses_from_output
from src.utils.metrics import ndcg_at_k, hit_rate_at_k, mrr_at_k
from tqdm import tqdm
from torch.amp import GradScaler


def probability(value):
    parsed = float(value)
    if not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError('probability must satisfy 0.0 <= value < 1.0')
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description='Integrated multimodal model training script')
    parser.add_argument('--emb_dim', type=int, default=768, help='input embedding dimension')
    parser.add_argument('--latent_dim', type=int, default=64, help='latent dimension')
    parser.add_argument('--enc_hidden', type=int, default=128)
    parser.add_argument('--fusion_hidden', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=1)
    parser.add_argument('--mamba_d_state', type=int, default=16, help='Mamba d_state')
    parser.add_argument('--mamba_d_conv', type=int, default=2, help='Mamba d_conv')
    parser.add_argument('--mamba_expand', type=int, default=1, help='Mamba expand')
    parser.add_argument('--drop_path', type=probability, default=0.0, help='Mamba stochastic depth rate')
    parser.add_argument('--dropout_rate', type=probability, default=0.2, help='SASRec Transformer dropout rate')
    parser.add_argument('--sasrec_hidden_units', type=int, default=64, help='SASRec hidden units')
    parser.add_argument('--fusion_method', type=str, default='poe', choices=['vae', 'poe'], help='fusion method')
    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--alpha_mi', type=float, default=0.1)
    parser.add_argument('--alpha_vae', type=float, default=0.5)
    parser.add_argument('--alpha_sas', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=256, help='train batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='eval/test batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--img_cont', type=int, default=9, help='number of image vectors')
    parser.add_argument('--text_cont', type=int, default=9, help='number of text vectors')
    parser.add_argument('--save_path', type=str, default='dimmrec_games_seq50.pth')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping max norm; <=0 disables')
    parser.add_argument('--resume_checkpoint', type=str, default='', help='checkpoint path to resume training')
    parser.add_argument('--dataset', type=str, default='games', help='dataset name')
    parser.add_argument('--data_dir', type=str, default='/dataset', help='dataset root dir')
    parser.add_argument('--text_file', type=str, help='text feature file path')
    parser.add_argument('--image_file', type=str, help='image feature file path')
    parser.add_argument('--sequence_file', type=str, help='sequence file path')
    return parser.parse_args()

def get_default_paths(args):
    """Build default dataset file paths."""
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    text_file = args.text_file if args.text_file else os.path.join(dataset_dir, 'text_features.npy')
    image_file = args.image_file if args.image_file else os.path.join(dataset_dir, 'image_features.npy')
    sequence_file = args.sequence_file if args.sequence_file else os.path.join(dataset_dir, 'sequences.txt')
    return (text_file, image_file, sequence_file)

def prepare_data(args):
    """Prepare required data on CPU side."""
    print('Preparing data...')
    text_file, image_file, sequence_file = get_default_paths(args)
    print('Loading training data...')
    train_loader, _, _, _ = create_dataloader(text_file=text_file, image_file=image_file, sequence_file=sequence_file, max_seq_len=args.max_seq_len, neg_ratio=1, batch_size=args.train_batch_size, mode='train', num_workers=0, pin_memory=True)
    save_dir = os.path.dirname(args.save_path)
    if save_dir and (not os.path.exists(save_dir)):
        os.makedirs(save_dir)
    print('Data preparation completed.')
    return {'train_loader': train_loader, 'text_file': text_file, 'image_file': image_file, 'sequence_file': sequence_file}


def extract_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    return checkpoint


def filter_model_state_dict(model, state_dict):
    current_state = model.state_dict()
    return {
        key: value
        for key, value in state_dict.items()
        if key in current_state and current_state[key].shape == value.shape
    }


def load_compatible_model(model, checkpoint):
    state_dict = extract_model_state_dict(checkpoint)
    filtered_state_dict = filter_model_state_dict(model, state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    if missing_keys:
        print(f'[WARN] Missing model keys: {len(missing_keys)}')
    if unexpected_keys:
        print(f'[WARN] Unexpected model keys: {len(unexpected_keys)}')
    skipped = len(state_dict) - len(filtered_state_dict)
    if skipped > 0:
        print(f'[WARN] Skipped incompatible model keys: {skipped}')
    return filtered_state_dict


def try_load_training_state(checkpoint, optimizer_main, optimizer_mi, scheduler, scaler):
    try:
        if 'optimizer_main_state_dict' in checkpoint:
            optimizer_main.load_state_dict(checkpoint['optimizer_main_state_dict'])
        if 'optimizer_mi_state_dict' in checkpoint:
            optimizer_mi.load_state_dict(checkpoint['optimizer_mi_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
    except (KeyError, ValueError, RuntimeError) as exc:
        print(f'[WARN] Training state was not loaded: {exc}')


def train(args, preprocessed_data):
    device = torch.device(args.device)
    log_path = args.save_path.replace('.pth', '.log')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f'Training Log for {args.dataset}\n')
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write('=' * 50 + '\n\n')
    print(f'Training log file: {log_path}')
    model = IntegratedMultimodalModel(emb_dim=args.emb_dim, enc_hidden=args.enc_hidden, latent_dim=args.latent_dim, fusion_hidden=args.fusion_hidden, num_heads=args.num_heads, transformer_layers=args.transformer_layers, drop_path=args.drop_path, dropout_rate=args.dropout_rate, max_seq_len=args.max_seq_len, num_blocks=args.num_blocks, sasrec_hidden_units=args.sasrec_hidden_units, img_cont=args.img_cont, text_cont=args.text_cont, mamba_d_state=args.mamba_d_state, mamba_d_conv=args.mamba_d_conv, mamba_expand=args.mamba_expand).to(device)
    special_params = list(model.mi_loss_module.parameters())
    special_param_ids = set((id(p) for p in special_params))
    default_params = [p for p in model.parameters() if id(p) not in special_param_ids]
    optimizer_main = optim.Adam([{'params': default_params}], lr=args.lr)
    mi_params = [{'params': model.mi_loss_module.parameters(), 'lr': args.lr}]
    optimizer_mi = optim.Adam(mi_params, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_main, mode='max', factor=0.8, patience=10, verbose=True, min_lr=2e-05)
    scaler = GradScaler() if torch.cuda.is_available() else GradScaler('cpu')
    start_epoch = 0
    best_ndcg = -float('inf')
    best_epoch = -1
    best_model_path = args.save_path.replace('.pth', '_best.pth')

    def _set_mi_requires_grad(enabled: bool):
        for p in model.mi_loss_module.parameters():
            p.requires_grad_(enabled)

    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f'Resuming training from checkpoint: {args.resume_checkpoint}')
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        load_compatible_model(model, checkpoint)
        if isinstance(checkpoint, dict):
            try_load_training_state(checkpoint, optimizer_main, optimizer_mi, scheduler, scaler)
        print(f'Resume succeeded! Start from epoch {start_epoch}.')
    print(f'Start training for {args.epochs} epochs on device: {device}')
    best_ndcg = -float('inf')
    best_epoch = -1
    best_model_path = args.save_path.replace('.pth', '_best.pth')
    val_interval = 2
    early_stop_patience = 10
    stale_val_count = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = len(preprocessed_data['train_loader'])
        for batch_idx, batch_data in enumerate(tqdm(preprocessed_data['train_loader'], desc=f'Epoch {epoch + 1}/{args.epochs}')):
            text_batch = batch_data['text'].to(device, non_blocking=True)
            img_batch = batch_data['image'].to(device, non_blocking=True)
            mask = batch_data['mask'].to(device, non_blocking=True)
            neg_text = batch_data['neg_text'].to(device, non_blocking=True) if 'neg_text' in batch_data else None
            neg_img = batch_data['neg_image'].to(device, non_blocking=True) if 'neg_image' in batch_data else None
            model_output = model(text_batch, img_batch, mask=mask, fusion_method=args.fusion_method, neg_text=neg_text, neg_img=neg_img)
            try:
                z_img = model_output['z_img'].detach()
                z_text = model_output['z_text'].detach()
                z_shared = model_output['z_shared'].detach()
            except KeyError:
                print('Error: missing keys in model_output')
                raise
            mi_learning_loss = model.mi_loss_module.learning_loss(z_img, z_text, z_shared, mask=mask)
            optimizer_mi.zero_grad(set_to_none=True)
            scaler.scale(mi_learning_loss).backward()
            scaler.step(optimizer_mi)
            scaler.update()
            _set_mi_requires_grad(False)
            total_loss, loss_dict = compute_losses_from_output(model=model, model_output=model_output, device=device, mask=mask, alpha_mi=args.alpha_mi, alpha_vae=args.alpha_vae, alpha_sas=args.alpha_sas)
            _set_mi_requires_grad(True)
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f'[WARN] NaN/Inf loss detected, skip this batch. loss={total_loss}')
                del text_batch, img_batch, mask, total_loss, loss_dict, model_output
                continue
            optimizer_main.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer_main)
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(default_params, max_norm=args.grad_clip)
            scaler.step(optimizer_main)
            scaler.update()
            optimizer_main.zero_grad(set_to_none=True)
            epoch_loss += total_loss.item()
            if batch_idx % 300 == 0:
                print(f'\nEpoch {epoch + 1}, Batch {batch_idx}:')
                print(f'  Total Loss: {total_loss.item():.4f}')
                print(f"  VAE Loss: {loss_dict['vae_loss']:.4f} (weighted: {loss_dict['weighted_vae']:.4f})")
                print(f"  MI Loss: {loss_dict['mi_loss']:.4f} (weighted: {loss_dict['weighted_mi']:.4f})")
                print(f"  SAS Loss: {loss_dict['sas_loss']:.4f} (weighted: {loss_dict['weighted_sas']:.4f})")
            del text_batch, img_batch, mask, total_loss, loss_dict, model_output
        avg_total_loss = epoch_loss / max(1, num_batches)
        print(f'\nEpoch {epoch + 1} : Total Loss: {avg_total_loss:.4f}')
        do_val = (epoch + 1) % val_interval == 0
        if do_val:
            print(f'\n>>> Epoch {epoch + 1}: running validation...')
            val_loader, _, _, _ = create_dataloader(text_file=preprocessed_data['text_file'], image_file=preprocessed_data['image_file'], sequence_file=preprocessed_data['sequence_file'], max_seq_len=args.max_seq_len, neg_ratio=100, batch_size=args.eval_batch_size, mode='val')
            val_metrics = evaluate(args, model=model, data_loader=val_loader)
            ndcg_score = val_metrics.get('ndcg@10', 0.0)
            is_best = ndcg_score > best_ndcg
            scheduler.step(ndcg_score)
            print(f'Val NDCG@10: {ndcg_score:.4f} | best: {best_ndcg:.4f} (epoch {best_epoch})')
            print(f"Current learning rate: {optimizer_main.param_groups[0]['lr']:.6f}")
            val_metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in val_metrics.items()])
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch + 1} Validation: {val_metrics_str}\n")
            if is_best:
                best_ndcg = ndcg_score
                best_epoch = epoch + 1
                stale_val_count = 0
                best_checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_main_state_dict': optimizer_main.state_dict(), 'optimizer_mi_state_dict': optimizer_mi.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict() if 'scaler' in locals() else None}
                torch.save(best_checkpoint, best_model_path)
                print(f'New best model at epoch {best_epoch}, NDCG@10 = {best_ndcg:.4f}')
            else:
                stale_val_count += 1
                print(f'No validation improvement: {stale_val_count}/{early_stop_patience}')
            del val_loader
            if stale_val_count >= early_stop_patience:
                print(f'Early stopping at epoch {epoch + 1}.')
                break
        else:
            is_best = False
            ndcg_score = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if best_epoch < 0:
        best_epoch = epoch + 1
        best_ndcg = float('nan')
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_main_state_dict': optimizer_main.state_dict(), 'optimizer_mi_state_dict': optimizer_mi.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict()}, best_model_path)
    print(f'\nTraining complete. Best model from epoch {best_epoch}, best NDCG@10 = {best_ndcg:.4f}')
    print('>>> Loading best model for final test...')
    ckpt = torch.load(best_model_path, map_location=device)
    load_compatible_model(model, ckpt)
    test_loader, _, _, _ = create_dataloader(text_file=preprocessed_data['text_file'], image_file=preprocessed_data['image_file'], sequence_file=preprocessed_data['sequence_file'], max_seq_len=args.max_seq_len, neg_ratio=100, batch_size=args.eval_batch_size, mode='test')
    final_metrics = evaluate(args, model=model, data_loader=test_loader)
    final_metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in final_metrics.items()])
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'=' * 20} FINAL TEST RESULT (Best Epoch {best_epoch}) {'=' * 20}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f'Metrics: {final_metrics_str}\n')
        f.write('=' * 60 + '\n')
    print(f'Final test results appended to log: {log_path}')
    del test_loader

def evaluate(args, model=None, data_loader=None):
    device = torch.device(args.device)
    if model is None:
        if not os.path.exists(args.save_path):
            print(f'Error: model file not found: {args.save_path}')
            return {}
        model = IntegratedMultimodalModel(emb_dim=args.emb_dim, enc_hidden=args.enc_hidden, latent_dim=args.latent_dim, fusion_hidden=args.fusion_hidden, num_heads=args.num_heads, transformer_layers=args.transformer_layers, drop_path=args.drop_path, dropout_rate=args.dropout_rate, max_seq_len=args.max_seq_len, num_blocks=args.num_blocks, sasrec_hidden_units=args.sasrec_hidden_units, img_cont=args.img_cont, text_cont=args.text_cont, mamba_d_state=args.mamba_d_state, mamba_d_conv=args.mamba_d_conv, mamba_expand=args.mamba_expand).to(device)
        checkpoint = torch.load(args.save_path, map_location=device)
        load_compatible_model(model, checkpoint)
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc='', leave=False)):
            text_batch = batch_data['text'].to(device)
            img_batch = batch_data['image'].to(device)
            mask = batch_data['mask'].to(device)
            model_output = model(text_batch, img_batch, mask=mask, fusion_method=args.fusion_method)
            predictions = model_output['sas_output']
            if predictions is None:
                continue
            pred_cpu = predictions.detach().cpu()
            for j in range(pred_cpu.size(0)):
                all_predictions.append(pred_cpu[j])
                all_labels.append(0)
            del text_batch, img_batch, model_output, predictions
    metrics_results = {}
    if all_predictions:
        scores_tensor = torch.stack(all_predictions, dim=0)
        labels_tensor = torch.tensor(all_labels)
        scores_tensor = scores_tensor.to(device)
        labels_tensor = labels_tensor.to(device)
        k_values = [5, 10]
        for k in k_values:
            actual_k = min(k, scores_tensor.shape[1])
            metrics_results[f'ndcg@{actual_k}'] = ndcg_at_k(scores_tensor, labels_tensor, actual_k)
            metrics_results[f'hit_rate@{actual_k}'] = hit_rate_at_k(scores_tensor, labels_tensor, actual_k)
            metrics_results[f'mrr@{actual_k}'] = mrr_at_k(scores_tensor, labels_tensor, actual_k)
        metrics_str = ' '.join([f'{metric}: {value:.4f}' for metric, value in metrics_results.items()])
        print(f'Metrics: {metrics_str}')
    return metrics_results

def main():
    """Main entrypoint."""
    print('\n=== Start ===')
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    preprocessed_data = prepare_data(args)
    print('\n=== Start Training ===')
    train(args, preprocessed_data)
if __name__ == '__main__':
    main()
