# DiMMRec

PyTorch implementation of **DiMMRec** for multimodal sequential recommendation.

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies:

- torch
- numpy
- tqdm
- mamba-ssm

## Repository Structure

```text
.
|-- train.py
|-- requirements.txt
|-- src/
|   |-- data/
|   |   `-- dataloader.py
|   |-- models/
|   |   |-- dimmrec.py
|   |   |-- vae.py
|   |   |-- mamba_block.py
|   |   |-- moe_fusion.py
|   |   |-- attention_fusion.py
|   |   |-- sasrec.py
|   |   `-- mutual_information.py
|   `-- utils/
|       `-- metrics.py
`-- README.md
```

## Data Preparation

This repository trains from processed multimodal feature files. 

The expected default data directory is:

```text
/dataset/<dataset>/
|-- text_features.npy
|-- image_features.npy
`-- sequences.txt
```

For example, with `--dataset games`, place processed files in:

```text
/dataset/games/
```

### Processed Feature Format

`text_features.npy` and `image_features.npy` should be NumPy-saved Python dictionaries:

```python
{
    item_id: feature_array
}
```

The feature arrays are expected to be multi-vector item features. With the default settings, both modalities use 9 vectors per item and 768-dimensional vectors:

```text
text feature:  [9, 768]
image feature: [9, 768]
```

If the number of vectors is changed, keep it consistent with:

```bash
--text_cont <num_text_vectors>
--img_cont <num_image_vectors>
```

Only item IDs appearing in both text and image feature dictionaries are used.

### Sequence Format

`sequences.txt` contains one chronological user sequence per line:

```text
item_id_1 item_id_2 item_id_3 ...
```

A typical preprocessing procedure is:

1. Sort each user's interactions by timestamp.
2. Write each user sequence as one line in `sequences.txt`.
3. Save item text and image features as the two `.npy` dictionaries above.

The dataloader uses the last two items of each sequence for evaluation:

```text
train: seq[:-2]
valid: seq[-2]
test:  seq[-1]
```

Training uses 1 negative sample per positive item. Validation and testing use 100 negative samples.

## Training

Run with the default data layout:

```bash
python train.py --dataset games
```

Use a custom data root:

```bash
python train.py \
  --data_dir /path/to/dataset_root \
  --dataset games
```

Use explicit files:

```bash
python train.py \
  --text_file /path/to/text_features.npy \
  --image_file /path/to/image_features.npy \
  --sequence_file /path/to/sequences.txt \
  --save_path dimmrec_games_seq50.pth
```

Common settings:

```bash
python train.py \
  --epochs 100 \
  --max_seq_len 50 \
  --train_batch_size 256 \
  --eval_batch_size 32 \
  --lr 0.0002 \
  --latent_dim 64 \
  --dropout_rate 0.2 \
  --device cuda
```

`--dropout_rate` controls the SASRec Transformer dropout searched in the
paper. `--drop_path` separately controls stochastic depth inside the Mamba
blocks and defaults to `0.0` because it is not part of the reported dropout
search.

Resume training:

```bash
python train.py \
  --resume_checkpoint dimmrec_games_seq50_best.pth \
  --save_path dimmrec_games_seq50.pth
```
