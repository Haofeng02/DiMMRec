# DiMMRec

PyTorch implementation of **DiMMRec** for multimodal sequential
recommendation.

## Requirements

Install the training dependencies:

```bash
pip install -r requirements.txt
```

Feature extraction additionally requires:

```bash
pip install -r requirements-preprocess.txt
```

## Data

Datasets are stored under `src/data/<dataset>/`:

```text
src/data/<dataset>/
|-- raw_texts.txt          # prepare locally
|-- image_urls.txt
|-- sequences.txt
|-- text_features.npy       # generated, not tracked by Git
`-- image_features.npy      # generated, not tracked by Git
```

The raw files use the following formats:

```text
raw_texts.txt:  <item_id> <raw text>
image_urls.txt: <item_id> <image URL>
sequences.txt:  <item_id_1> <item_id_2> ...
```

The repository includes `image_urls.txt` and `sequences.txt` for Amazon Games
and Amazon Baby. Download the `Video_Games` and `Baby_Products` review and
metadata files from [Amazon Reviews'23](https://amazon-reviews-2023.github.io/)
and prepare `raw_texts.txt` locally. The three input files must use the same
integer item-ID mapping. If a different mapping is used, regenerate all three
files together. `raw_texts.txt` is ignored by Git because of its size.

MicroLens-50K uses a different source format; download it from the
[official dataset page](https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/)
and convert its text and visual content to the feature format below. The
repository currently includes its processed interaction sequences only.

## Feature Extraction

After preparing all three Amazon input files, generate text and image features
with:

```bash
python scripts/prepare_amazon_features.py \
  --dataset-dir src/data/games \
  --modalities both \
  --device cuda
```

Replace `games` with `baby` for Amazon Baby. The default pipeline uses an LLM
to produce salience-ordered keywords, encodes the first nine keywords
individually with SBERT, and extracts nine L2-normalized DINOv2 region features
using 3 x 3 adaptive average pooling.

The generated NumPy files are dictionaries of item features:

```python
{
    item_id: feature_array  # shape: [9, 768]
}
```

## Training and Evaluation

Train with the default repository layout:

```bash
python train.py \
  --dataset games \
  --device cuda \
  --save_path dimmrec_games_seq50.pth
```

Use explicit feature files when needed:

```bash
python train.py \
  --text_file /path/to/text_features.npy \
  --image_file /path/to/image_features.npy \
  --sequence_file /path/to/sequences.txt \
  --save_path dimmrec.pth
```

Resume training with:

```bash
python train.py \
  --resume_checkpoint dimmrec_games_seq50_best.pth \
  --save_path dimmrec_games_seq50.pth
```

