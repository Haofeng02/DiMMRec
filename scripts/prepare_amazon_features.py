"""Build DiMMRec feature dictionaries from Amazon-style raw data files."""

import argparse
import hashlib
import io
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (  # noqa: E402
    DataFormatError,
    load_cached_feature,
    load_keyword_file,
    merge_feature_cache,
    pad_feature_vectors,
    parse_keyword_list,
    pool_patch_tokens,
    save_cached_feature,
    validate_amazon_inputs,
)


KEYWORD_COUNT = 9
KEYWORD_PROMPT = """You are a keyword generation assistant for item recommendation.
Given the raw product description below, generate 20 to 30 concise English keywords or short keyphrases. Infer useful attributes when supported by the description. Remove duplicates, order the keywords from most to least salient, and return exactly one comma-and-space separated line.

Raw Description:
{description}

Format:
Keywords: keyword1, keyword2, keyword3"""


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build DiMMRec features from Amazon-style raw files"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--modalities",
        choices=("text", "image", "both"),
        default="both",
    )
    parser.add_argument(
        "--text-model",
        default="sentence-transformers/all-mpnet-base-v2",
    )
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--keywords-file",
        type=Path,
        default=Path("keywords.txt"),
        help="existing/generated '<item_id> keyword1, keyword2, ...' file",
    )
    parser.add_argument("--llm-batch-size", type=int, default=4)
    parser.add_argument("--llm-max-input-tokens", type=int, default=2048)
    parser.add_argument("--llm-max-new-tokens", type=int, default=192)
    parser.add_argument("--image-model", default="facebook/dinov2-base")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", type=Path, default=Path(".feature_cache"))
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_device(requested):
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def encode_text_item(keywords, model, batch_size=32):
    selected = keywords[:KEYWORD_COUNT]
    if len(selected) != KEYWORD_COUNT:
        raise DataFormatError(
            f"expected at least {KEYWORD_COUNT} keywords, got {len(selected)}"
        )
    vectors = model.encode(
        selected,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return pad_feature_vectors(vectors)


def _model_cache_key(model_name):
    return hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:10]


def _keyword_cache_key(keywords, item_ids):
    digest = hashlib.sha256()
    for item_id in item_ids:
        digest.update(str(item_id).encode("ascii"))
        digest.update(b"\0")
        digest.update(", ".join(keywords[item_id][:KEYWORD_COUNT]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:10]


def _clean_generated_keywords(generated):
    generated = generated.strip().strip("`")
    marker = generated.casefold().find("keywords:")
    if marker >= 0:
        generated = generated[marker + len("keywords:") :]
    generated = generated.replace("\r", " ").replace("\n", ", ")
    return parse_keyword_list(generated, minimum_count=KEYWORD_COUNT)[:30]


def _load_cached_keywords(cache_dir, item_id):
    path = cache_dir / f"{item_id}.txt"
    if not path.is_file():
        return None
    try:
        return parse_keyword_list(
            path.read_text(encoding="utf-8"),
            minimum_count=KEYWORD_COUNT,
        )[:30]
    except DataFormatError:
        return None


def _save_cached_keywords(cache_dir, item_id, keywords):
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{item_id}.txt"
    temporary = cache_dir / f".{item_id}.tmp"
    temporary.write_text(", ".join(keywords), encoding="utf-8")
    temporary.replace(target)


def _write_keyword_file(path, keywords, item_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for item_id in item_ids:
            handle.write(f"{item_id} {', '.join(keywords[item_id])}\n")
    temporary.replace(path)


def load_or_generate_keywords(
    texts,
    item_ids,
    keywords_path,
    cache_dir,
    model_name,
    device,
    batch_size,
    max_input_tokens,
    max_new_tokens,
):
    if keywords_path.is_file():
        print(f"keywords: loading {keywords_path}")
        return load_keyword_file(
            keywords_path,
            item_ids,
            minimum_count=KEYWORD_COUNT,
        )

    keyword_cache = cache_dir / f"keywords-{_model_cache_key(model_name)}"
    keywords = {}
    pending = []
    for item_id in item_ids:
        cached = _load_cached_keywords(keyword_cache, item_id)
        if cached is None:
            pending.append(item_id)
        else:
            keywords[item_id] = cached

    if pending:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"keywords: generating {len(pending)} profiles with {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
        ).to(device).eval()

        completed = len(item_ids) - len(pending)
        for start in range(0, len(pending), batch_size):
            batch_ids = pending[start : start + batch_size]
            prompts = []
            for item_id in batch_ids:
                messages = [
                    {
                        "role": "system",
                        "content": "Return only the requested keyword line.",
                    },
                    {
                        "role": "user",
                        "content": KEYWORD_PROMPT.format(
                            description=texts[item_id]
                        ),
                    },
                ]
                prompts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            prompt_width = encoded["input_ids"].shape[1]
            with torch.inference_mode():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated_rows = tokenizer.batch_decode(
                outputs[:, prompt_width:],
                skip_special_tokens=True,
            )
            for item_id, generated in zip(batch_ids, generated_rows):
                try:
                    profile = _clean_generated_keywords(generated)
                except DataFormatError as exc:
                    raise DataFormatError(
                        f"item {item_id}: Qwen returned an invalid keyword profile; "
                        "rerun to retry or provide --keywords-file"
                    ) from exc
                keywords[item_id] = profile
                _save_cached_keywords(keyword_cache, item_id, profile)
            completed += len(batch_ids)
            print(f"keywords: {completed}/{len(item_ids)}")

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    _write_keyword_file(keywords_path, keywords, item_ids)
    print(f"keywords: wrote {keywords_path}")
    return keywords


def process_text_features(
    keywords,
    item_ids,
    cache_dir,
    output_path,
    model_name,
    batch_size,
    device,
    overwrite,
):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    dimension = model.get_sentence_embedding_dimension()
    if dimension != 768:
        raise DataFormatError(f"text model dimension must be 768, got {dimension}")

    text_cache = cache_dir / (
        f"text-keywords9-{_model_cache_key(model_name)}-"
        f"{_keyword_cache_key(keywords, item_ids)}"
    )
    pending = [
        item_id
        for item_id in item_ids
        if load_cached_feature(text_cache, item_id) is None
    ]
    completed = len(item_ids) - len(pending)
    for start in range(0, len(pending), batch_size):
        batch_ids = pending[start : start + batch_size]
        flat_keywords = [
            keyword
            for item_id in batch_ids
            for keyword in keywords[item_id][:KEYWORD_COUNT]
        ]
        flat_vectors = model.encode(
            flat_keywords,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        for offset, item_id in enumerate(batch_ids):
            start_index = offset * KEYWORD_COUNT
            feature = pad_feature_vectors(
                flat_vectors[start_index : start_index + KEYWORD_COUNT]
            )
            save_cached_feature(text_cache, item_id, feature)
        completed += len(batch_ids)
        print(f"text: {completed}/{len(item_ids)}")
    merge_feature_cache(text_cache, item_ids, output_path, overwrite=overwrite)


def download_image(session, url, retries, timeout):
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"download failed after {retries + 1} attempts: {last_error}"
    ) from last_error


def encode_image_item(image, processor, model, device):
    import numpy as np
    import torch
    import torch.nn.functional as F

    inputs = processor(images=image, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        pooled = pool_patch_tokens(patch_tokens)
        pooled = F.normalize(pooled, p=2, dim=-1)
    feature = pooled[0].detach().cpu().numpy().astype(np.float32, copy=False)
    if feature.shape != (9, 768):
        raise DataFormatError(
            f"image model output must be [9, 768], got {feature.shape}"
        )
    return feature


def process_image_features(
    image_urls,
    item_ids,
    cache_dir,
    output_path,
    model_name,
    device,
    retries,
    timeout,
    overwrite,
):
    import requests
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    if hidden_size != 768:
        raise DataFormatError(
            f"image model dimension must be 768, got {hidden_size}"
        )

    image_cache = cache_dir / (
        f"image-dinov2-pool3-l2-{_model_cache_key(model_name)}"
    )
    failure_path = cache_dir / "image_failures.tsv"
    failures = []
    with requests.Session() as session:
        session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; DiMMRecPreprocessor/1.0)"}
        )
        for position, item_id in enumerate(item_ids, start=1):
            if load_cached_feature(image_cache, item_id) is None:
                url = image_urls[item_id]
                try:
                    payload = download_image(
                        session,
                        url,
                        retries=retries,
                        timeout=timeout,
                    )
                    with Image.open(io.BytesIO(payload)) as opened:
                        image = opened.convert("RGB")
                    feature = encode_image_item(image, processor, model, device)
                    save_cached_feature(image_cache, item_id, feature)
                except Exception as exc:
                    message = (
                        str(exc)
                        .replace("\t", " ")
                        .replace("\r", " ")
                        .replace("\n", " ")
                    )
                    failures.append(
                        (item_id, url, type(exc).__name__, message)
                    )
            if position % 100 == 0 or position == len(item_ids):
                print(f"image: {position}/{len(item_ids)}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    if failures:
        with failure_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("item_id\turl\terror_type\tmessage\n")
            for row in failures:
                handle.write("\t".join(map(str, row)) + "\n")
        raise DataFormatError(
            f"{len(failures)} images failed; see {failure_path}"
        )
    if failure_path.exists():
        failure_path.unlink()
    merge_feature_cache(image_cache, item_ids, output_path, overwrite=overwrite)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (
        args.batch_size <= 0
        or args.llm_batch_size <= 0
        or args.llm_max_input_tokens <= 0
        or args.llm_max_new_tokens <= 0
        or args.retries < 0
        or args.timeout <= 0
    ):
        print(
            "error: batch sizes, token limits, and timeout must be positive; "
            "retries must be nonnegative",
            file=sys.stderr,
        )
        return 2

    dataset_dir = args.dataset_dir.resolve()
    cache_dir = (
        args.cache_dir
        if args.cache_dir.is_absolute()
        else dataset_dir / args.cache_dir
    )
    keywords_path = (
        args.keywords_file
        if args.keywords_file.is_absolute()
        else dataset_dir / args.keywords_file
    )
    try:
        texts, image_urls, item_ids = validate_amazon_inputs(dataset_dir)
        device = resolve_device(args.device)
        if args.modalities in ("text", "both"):
            keywords = load_or_generate_keywords(
                texts,
                item_ids,
                keywords_path,
                cache_dir,
                args.llm_model,
                device,
                args.llm_batch_size,
                args.llm_max_input_tokens,
                args.llm_max_new_tokens,
            )
            process_text_features(
                keywords,
                item_ids,
                cache_dir,
                dataset_dir / "text_features.npy",
                args.text_model,
                args.batch_size,
                device,
                args.overwrite,
            )
        if args.modalities in ("image", "both"):
            process_image_features(
                image_urls,
                item_ids,
                cache_dir,
                dataset_dir / "image_features.npy",
                args.image_model,
                device,
                args.retries,
                args.timeout,
                args.overwrite,
            )
    except (DataFormatError, FileExistsError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
