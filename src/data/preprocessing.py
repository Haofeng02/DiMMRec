import math
from pathlib import Path
from typing import Dict, List, Set, Tuple


NUM_VECTORS = 9
EMBED_DIM = 768


class DataFormatError(ValueError):
    """Raised when raw dataset files violate the documented format."""


def parse_id_value_file(path: Path) -> Dict[int, str]:
    values: Dict[int, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            parts = line.split(" ", 1)
            if len(parts) != 2 or not parts[1].strip():
                raise DataFormatError(f"{path}:{line_number}: expected '<item_id> <value>'")
            try:
                item_id = int(parts[0])
            except ValueError as exc:
                raise DataFormatError(
                    f"{path}:{line_number}: invalid item ID {parts[0]!r}"
                ) from exc
            if item_id in values:
                raise DataFormatError(
                    f"{path}:{line_number}: duplicate item ID {item_id}"
                )
            values[item_id] = parts[1].strip()
    return values


def read_sequence_item_ids(path: Path) -> Set[int]:
    item_ids: Set[int] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            for token in raw_line.split():
                try:
                    item_ids.add(int(token))
                except ValueError as exc:
                    raise DataFormatError(
                        f"{path}:{line_number}: invalid sequence item ID {token!r}"
                    ) from exc
    return item_ids


def validate_amazon_inputs(
    dataset_dir: Path,
) -> Tuple[Dict[int, str], Dict[int, str], List[int]]:
    dataset_dir = Path(dataset_dir)
    text_path = dataset_dir / "raw_texts.txt"
    image_path = dataset_dir / "image_urls.txt"
    sequence_path = dataset_dir / "sequences.txt"
    missing = [
        path.name
        for path in (text_path, image_path, sequence_path)
        if not path.is_file()
    ]
    if missing:
        raise DataFormatError(f"missing required files: {', '.join(missing)}")

    texts = parse_id_value_file(text_path)
    image_urls = parse_id_value_file(image_path)
    sequence_ids = read_sequence_item_ids(sequence_path)
    text_ids = set(texts)
    image_ids = set(image_urls)
    if text_ids != image_ids or text_ids != sequence_ids:
        raise DataFormatError(
            "ID sets do not match: "
            f"text={len(text_ids)}, image={len(image_ids)}, "
            f"sequence={len(sequence_ids)}, "
            f"missing_text={len(sequence_ids - text_ids)}, "
            f"missing_image={len(sequence_ids - image_ids)}, "
            f"unused_text={len(text_ids - sequence_ids)}, "
            f"unused_image={len(image_ids - sequence_ids)}"
        )
    return texts, image_urls, sorted(sequence_ids)


def parse_keyword_list(value, minimum_count=9):
    """Parse a salience-ordered, comma-separated keyword line."""
    keywords = []
    seen = set()
    for part in value.split(","):
        keyword = part.strip()
        normalized = keyword.casefold()
        if keyword and normalized not in seen:
            keywords.append(keyword)
            seen.add(normalized)
    if len(keywords) < minimum_count:
        raise DataFormatError(
            f"expected at least {minimum_count} unique keywords, got {len(keywords)}"
        )
    return keywords


def load_keyword_file(path, expected_item_ids, minimum_count=9):
    """Load and validate keyword profiles for all expected items."""
    raw_keywords = parse_id_value_file(Path(path))
    expected = set(expected_item_ids)
    actual = set(raw_keywords)
    if actual != expected:
        raise DataFormatError(
            "keyword ID set does not match dataset: "
            f"missing={len(expected - actual)}, extra={len(actual - expected)}"
        )
    return {
        item_id: parse_keyword_list(raw_keywords[item_id], minimum_count)
        for item_id in sorted(expected)
    }


def pad_feature_vectors(
    vectors,
    num_vectors=NUM_VECTORS,
    embed_dim=EMBED_DIM,
):
    import numpy as np

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != embed_dim or array.shape[0] > num_vectors:
        raise DataFormatError(
            f"expected at most {num_vectors} vectors of dimension {embed_dim}, "
            f"got {array.shape}"
        )
    output = np.zeros((num_vectors, embed_dim), dtype=np.float32)
    output[: array.shape[0]] = array
    return output


def pool_patch_tokens(patch_tokens, grid_size=3):
    import torch.nn.functional as F

    if patch_tokens.ndim != 3:
        raise DataFormatError(
            f"expected [batch, patches, dim], got {tuple(patch_tokens.shape)}"
        )
    batch, patch_count, dim = patch_tokens.shape
    side = math.isqrt(patch_count)
    if side * side != patch_count:
        raise DataFormatError(f"patch count {patch_count} is not a square grid")
    grid = patch_tokens.transpose(1, 2).reshape(batch, dim, side, side)
    pooled = F.adaptive_avg_pool2d(grid, (grid_size, grid_size))
    return pooled.flatten(2).transpose(1, 2).contiguous()


def _validate_feature(array):
    import numpy as np

    array = np.asarray(array)
    return array.shape == (NUM_VECTORS, EMBED_DIM) and array.dtype == np.float32


def save_cached_feature(cache_dir, item_id, feature):
    import numpy as np

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    array = np.asarray(feature, dtype=np.float32)
    if not _validate_feature(array):
        raise DataFormatError(
            f"invalid feature for item {item_id}: {array.shape}, {array.dtype}"
        )
    target = cache_dir / f"{item_id}.npy"
    temporary = cache_dir / f".{item_id}.tmp"
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(target)


def load_cached_feature(cache_dir, item_id):
    import numpy as np

    path = Path(cache_dir) / f"{item_id}.npy"
    if not path.is_file():
        return None
    array = np.load(path, allow_pickle=False)
    return array if _validate_feature(array) else None


def merge_feature_cache(cache_dir, item_ids, output_path, overwrite=False):
    import numpy as np

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {output_path}; pass --overwrite"
        )
    features = {}
    for item_id in item_ids:
        feature = load_cached_feature(cache_dir, item_id)
        if feature is None:
            raise DataFormatError(
                f"missing or invalid cached feature for item {item_id}"
            )
        features[item_id] = feature
    with output_path.open("wb") as handle:
        np.save(handle, features, allow_pickle=True)
