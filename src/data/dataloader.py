import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MultimodalSequenceDataset(Dataset):
    def __init__(self, item_ids: List[int], text_features: torch.Tensor, image_features: torch.Tensor, sequences: List[List[int]], max_seq_len: int=10, neg_ratio: int=1, mode: str='train'):
        self.max_seq_len = max_seq_len
        self.neg_ratio = neg_ratio
        self.mode = mode
        random.seed(42)

        zero_text = torch.zeros(1, text_features.shape[1], text_features.shape[2])
        zero_image = torch.zeros(1, image_features.shape[1], image_features.shape[2])
        self.text_features = torch.cat([zero_text, text_features], dim=0)
        self.image_features = torch.cat([zero_image, image_features], dim=0)
        self.item_to_index = {iid: idx + 1 for idx, iid in enumerate(item_ids)}
        self.item_ids = item_ids
        self.num_items = len(item_ids) + 1
        self.sequences = [seq[-(max_seq_len + 3):] for seq in sequences if len(seq) > 1]

        if self.mode == 'train':
            self.samples = self._generate_train_sequences()
            self.total_len = max_seq_len + 1
        else:
            self.samples = self._generate_eval_samples()
            self.total_len = max_seq_len + 1 + neg_ratio

    def __len__(self):
        return len(self.samples)

    def _generate_train_sequences(self):
        samples = []
        for seq in self.sequences:
            train_seq = seq[:-2]
            if len(train_seq) >= 2:
                samples.append(train_seq)
        print(f'[Train] samples: {len(samples)}')
        return samples

    def _generate_eval_samples(self):
        samples = []
        for seq in self.sequences:
            if len(seq) < 2:
                continue
            if self.mode == 'val':
                hist_seq = seq[:-2]
                pos_item = seq[-2]
            else:
                hist_seq = seq[:-1]
                pos_item = seq[-1]
            samples.append((hist_seq, pos_item))
        print(f'[{self.mode.upper()}] samples: {len(samples)}')
        return samples

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self._get_train_item(idx)
        return self._get_eval_item(idx)

    def _get_train_item(self, idx):
        train_seq = self.samples[idx]
        train_seq = [iid for iid in train_seq if iid in self.item_to_index]
        if len(train_seq) < self.total_len:
            train_seq = [0] * (self.total_len - len(train_seq)) + train_seq
        else:
            train_seq = train_seq[-self.total_len:]

        indices = [self.item_to_index.get(iid, 0) for iid in train_seq]
        text_feats = self.text_features[indices]
        image_feats = self.image_features[indices]
        neg_indices = self._sample_train_negatives(train_seq)
        neg_text_feats = self.text_features[neg_indices]
        neg_image_feats = self.image_features[neg_indices]
        return {
            'text_features': text_feats,
            'image_features': image_feats,
            'neg_text_features': neg_text_feats,
            'neg_image_features': neg_image_feats,
        }

    def _sample_train_negatives(self, train_seq):
        per_step_neg_ids = []
        for t in range(self.max_seq_len):
            pos_item = train_seq[t + 1]
            if pos_item == 0:
                per_step_neg_ids.append([0] * self.neg_ratio)
                continue
            hist_prefix = {iid for iid in train_seq[:t + 1] if iid != 0}
            neg_items = []
            for _ in range(self.neg_ratio):
                neg_item = random.choice(self.item_ids)
                retry = 0
                while (neg_item == pos_item or neg_item in hist_prefix) and retry < 100:
                    neg_item = random.choice(self.item_ids)
                    retry += 1
                if neg_item == pos_item or neg_item in hist_prefix:
                    neg_item = 0
                neg_items.append(neg_item)
            per_step_neg_ids.append(neg_items)
        return [[self.item_to_index.get(iid, 0) for iid in row] for row in per_step_neg_ids]

    def _get_eval_item(self, idx):
        hist_seq, pos_item = self.samples[idx]
        hist_seq = [iid for iid in hist_seq if iid in self.item_to_index]
        if pos_item not in self.item_to_index:
            pos_item = 0

        if len(hist_seq) < self.max_seq_len:
            hist_seq = [0] * (self.max_seq_len - len(hist_seq)) + hist_seq
        else:
            hist_seq = hist_seq[-self.max_seq_len:]

        neg_items = []
        for _ in range(self.neg_ratio):
            neg_item = random.choice(self.item_ids)
            while neg_item == pos_item or neg_item in hist_seq:
                neg_item = random.choice(self.item_ids)
            neg_items.append(neg_item)

        all_item_ids = hist_seq + [pos_item] + neg_items
        if len(all_item_ids) < self.total_len:
            all_item_ids = [0] * (self.total_len - len(all_item_ids)) + all_item_ids
        elif len(all_item_ids) > self.total_len:
            all_item_ids = all_item_ids[-self.total_len:]

        indices = [self.item_to_index.get(iid, 0) for iid in all_item_ids]
        return {
            'text_features': self.text_features[indices],
            'image_features': self.image_features[indices],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    text_features = torch.stack([b['text_features'] for b in batch])
    image_features = torch.stack([b['image_features'] for b in batch])
    mask = text_features.abs().sum(dim=-1).sum(dim=-1) > 0
    out = {'text': text_features, 'image': image_features, 'mask': mask}
    if 'neg_text_features' in batch[0] and 'neg_image_features' in batch[0]:
        out['neg_text'] = torch.stack([b['neg_text_features'] for b in batch])
        out['neg_image'] = torch.stack([b['neg_image_features'] for b in batch])
    return out


def load_item_features(text_file: str, image_file: str) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    text_data = np.load(text_file, allow_pickle=True).item()
    image_data = np.load(image_file, allow_pickle=True).item()
    item_ids = sorted(set(text_data.keys()) & set(image_data.keys()))
    text_features = torch.from_numpy(np.array([text_data[iid] for iid in item_ids])).float()
    image_features = torch.from_numpy(np.array([image_data[iid] for iid in item_ids])).float()
    return item_ids, text_features, image_features


def load_sequences(sequence_file: str) -> List[List[int]]:
    sequences = []
    with open(sequence_file, 'r', encoding='utf-8') as f:
        for line in f:
            sequences.append(list(map(int, line.strip().split())))
    return sequences


def create_dataloader(text_file, image_file, sequence_file, max_seq_len=10, neg_ratio=1, batch_size=16, mode='train', shuffle=True, num_workers=0, pin_memory=False):
    item_ids, text_features, image_features = load_item_features(text_file, image_file)
    sequences = load_sequences(sequence_file)
    dataset = MultimodalSequenceDataset(item_ids, text_features, image_features, sequences, max_seq_len, neg_ratio, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True if num_workers > 0 else False, prefetch_factor=4 if num_workers > 0 else None)
    return dataloader, item_ids, text_features, image_features
