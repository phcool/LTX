"""
Download a filtered subset of HuggingFaceFV/finevideo dataset.

Two-step workflow (recommended):
    # Step 1: Build a local metadata index once (~10-20 min, no video bytes downloaded)
    python download_finevideo.py --build-index

    # Step 2: Query the index and list categories (instant)
    python download_finevideo.py --list-categories

    # Step 3: Download videos matching a category (only fetches matching rows)
    python download_finevideo.py --category Sports --max-videos 200

One-shot (skips index, slower):
    python download_finevideo.py --category Sports --max-videos 200 --no-index
"""

import argparse
import json
import os
import sys


REPO_ID = "HuggingFaceFV/finevideo"
INDEX_FILENAME = "finevideo_index.jsonl"


def default_output_dir() -> str:
    hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    return os.path.join(hf_home, "datasets", "finevideo")


def index_path(output_dir: str) -> str:
    return os.path.join(output_dir, INDEX_FILENAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download filtered FineVideo subset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root directory for videos, metadata, and index. "
             "Defaults to $HF_HOME/datasets/finevideo.",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--build-index",
        action="store_true",
        help="Download only the json column from all parquet files and save a local index. "
             "Run this once before --list-categories or filtering.",
    )
    mode.add_argument(
        "--list-categories",
        action="store_true",
        help="Print all categories from the local index (requires --build-index first).",
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by content_parent_category (e.g. 'Sports')",
    )
    parser.add_argument(
        "--fine-category",
        type=str,
        default=None,
        help="Filter by content_fine_category (e.g. 'Soccer')",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default=None,
        help="Filter by keyword in video description (case-insensitive)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=200,
        help="Maximum number of videos to download (default: 200)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=5.0,
        help="Minimum video duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
        help="Maximum video duration in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Skip the local index and stream directly (slower, no --build-index needed)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def build_index(output_dir: str) -> None:
    """Download only json column from all parquet shards and save a local JSONL index."""
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
    except ImportError:
        print("ERROR: install huggingface_hub and pyarrow:  pip install huggingface_hub pyarrow")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    out_path = index_path(output_dir)

    fs = HfFileSystem()
    parquet_files = sorted(fs.glob(f"datasets/{REPO_ID}/data/train-*.parquet"))
    total = len(parquet_files)
    print(f"Found {total} parquet shards. Downloading json column only...")
    print(f"Index will be saved to: {out_path}\n")

    written = 0
    with open(out_path, "w") as fout:
        for i, pq_path in enumerate(parquet_files):
            print(f"  [{i+1}/{total}] {os.path.basename(pq_path)}", end="", flush=True)
            try:
                with fs.open(pq_path, "rb") as f:
                    table = pq.read_table(f, columns=["json"])
                rows = table.to_pylist()
                for row in rows:
                    meta = row.get("json", {})
                    if not isinstance(meta, dict):
                        try:
                            meta = json.loads(meta)
                        except Exception:
                            continue
                    # Store a compact record: only the fields we need for filtering
                    record = {
                        "shard": os.path.basename(pq_path),
                        "parent_category": meta.get("content_parent_category", ""),
                        "fine_category": meta.get("content_fine_category", ""),
                        "duration": meta.get("duration_seconds", 0),
                        "description": meta.get("content_metadata", {}).get("description", ""),
                        "youtube_id": meta.get("youtube_id", ""),
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                print(f"  → {len(rows)} rows", flush=True)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)

    print(f"\nIndex built: {written} entries → {out_path}")
    print("Now run:  python download_finevideo.py --list-categories")


def load_index(output_dir: str) -> list[dict]:
    path = index_path(output_dir)
    if not os.path.exists(path):
        print(f"ERROR: Index not found at {path}")
        print("Run first:  python download_finevideo.py --build-index")
        sys.exit(1)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def list_categories(output_dir: str) -> None:
    records = load_index(output_dir)
    parent_cats: dict[str, int] = {}
    fine_cats: dict[str, int] = {}
    for r in records:
        p = r.get("parent_category", "")
        fi = r.get("fine_category", "")
        if p:
            parent_cats[p] = parent_cats.get(p, 0) + 1
        if fi:
            fine_cats[fi] = fine_cats.get(fi, 0) + 1

    print(f"\n=== Parent categories ({len(parent_cats)} total) ===")
    for c, n in sorted(parent_cats.items(), key=lambda x: -x[1]):
        print(f"  {n:>6}  {c}")
    print(f"\n=== Fine categories ({len(fine_cats)} total, top 60) ===")
    for c, n in sorted(fine_cats.items(), key=lambda x: -x[1])[:60]:
        print(f"  {n:>6}  {c}")


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def matches_filters(
    record: dict,
    category: str | None,
    fine_category: str | None,
    keyword: str | None,
    min_duration: float,
    max_duration: float,
) -> bool:
    duration = record.get("duration", 0)
    if not (min_duration <= duration <= max_duration):
        return False
    if category and record.get("parent_category", "") != category:
        return False
    if fine_category and record.get("fine_category", "") != fine_category:
        return False
    if keyword and keyword.lower() not in record.get("description", "").lower():
        return False
    return True


def matches_filters_meta(
    meta: dict,
    category: str | None,
    fine_category: str | None,
    keyword: str | None,
    min_duration: float,
    max_duration: float,
) -> bool:
    """Same logic but for raw sample json (used in streaming/no-index mode)."""
    return matches_filters(
        {
            "parent_category": meta.get("content_parent_category", ""),
            "fine_category": meta.get("content_fine_category", ""),
            "duration": meta.get("duration_seconds", 0),
            "description": meta.get("content_metadata", {}).get("description", ""),
        },
        category, fine_category, keyword, min_duration, max_duration,
    )


# ---------------------------------------------------------------------------
# Download via index (fast: only fetch matching parquet rows)
# ---------------------------------------------------------------------------

def download_via_index(args: argparse.Namespace, output_dir: str) -> None:
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
    except ImportError:
        print("ERROR: install huggingface_hub and pyarrow:  pip install huggingface_hub pyarrow")
        sys.exit(1)

    records = load_index(output_dir)

    matched = [
        r for r in records
        if matches_filters(r, args.category, args.fine_category, args.keyword,
                           args.min_duration, args.max_duration)
    ]
    print(f"Index has {len(records)} entries. Matched filter: {len(matched)}")
    if not matched:
        print("No matching entries. Try different filters or --list-categories.")
        return

    matched = matched[:args.max_videos]
    print(f"Will download {len(matched)} videos.\n")

    videos_dir = os.path.join(output_dir, "videos")
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # Group by shard to minimise number of remote file opens
    from collections import defaultdict
    shard_map: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, r in enumerate(matched):
        shard_map[r["shard"]].append((idx, r))

    fs = HfFileSystem()
    dataset_records: list[dict] = []
    downloaded = 0

    for shard_name, items in shard_map.items():
        shard_path = f"datasets/{REPO_ID}/data/{shard_name}"
        target_ids = {r["youtube_id"] for _, r in items}
        print(f"Opening shard {shard_name} for {len(items)} video(s)...")
        try:
            with fs.open(shard_path, "rb") as f:
                table = pq.read_table(f, columns=["json", "mp4"])
            rows = table.to_pylist()
        except Exception as e:
            print(f"  ERROR reading shard: {e}")
            continue

        # Build youtube_id → row map for fast lookup
        row_map: dict[str, dict] = {}
        for row in rows:
            meta = row.get("json", {})
            if not isinstance(meta, dict):
                try:
                    meta = json.loads(meta)
                except Exception:
                    continue
            yid = meta.get("youtube_id", "")
            if yid in target_ids:
                row_map[yid] = {"meta": meta, "mp4": row.get("mp4")}

        for idx, record in items:
            yid = record["youtube_id"]
            entry = row_map.get(yid)
            if not entry or not entry["mp4"]:
                print(f"  [{downloaded+1}/{len(matched)}] SKIP {yid} (not found in shard)")
                continue

            meta = entry["meta"]
            video_filename = f"video_{downloaded:05d}.mp4"
            video_path = os.path.join(videos_dir, video_filename)

            with open(video_path, "wb") as f:
                f.write(entry["mp4"])

            with open(os.path.join(metadata_dir, f"video_{downloaded:05d}.json"), "w") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            description = meta.get("content_metadata", {}).get("description", "")
            dataset_records.append({"caption": description, "media_path": video_path})

            downloaded += 1
            print(f"  [{downloaded}/{len(matched)}] {video_filename} "
                  f"({meta.get('duration_seconds', '?')}s) - {meta.get('content_fine_category', '?')}")

    _save_dataset_json(output_dir, dataset_records, downloaded)


# ---------------------------------------------------------------------------
# Download via streaming (slow, no index needed)
# ---------------------------------------------------------------------------

def download_via_streaming(args: argparse.Namespace, output_dir: str) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found. Run: pip install datasets")
        sys.exit(1)

    videos_dir = os.path.join(output_dir, "videos")
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    print("Loading FineVideo in streaming mode (this is slow without an index)...")
    dataset = load_dataset(REPO_ID, split="train", streaming=True)

    dataset_records: list[dict] = []
    downloaded = 0
    scanned = 0

    for sample in dataset:
        scanned += 1
        meta = sample.get("json", {})
        if not isinstance(meta, dict):
            try:
                meta = json.loads(meta)
            except Exception:
                continue

        if not matches_filters_meta(meta, args.category, args.fine_category,
                                    args.keyword, args.min_duration, args.max_duration):
            if scanned % 200 == 0:
                print(f"  scanned {scanned}, downloaded {downloaded}...", flush=True)
            continue

        video_bytes = sample.get("mp4")
        if not video_bytes:
            continue

        video_filename = f"video_{downloaded:05d}.mp4"
        video_path = os.path.join(videos_dir, video_filename)
        with open(video_path, "wb") as f:
            f.write(video_bytes)
        with open(os.path.join(metadata_dir, f"video_{downloaded:05d}.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        description = meta.get("content_metadata", {}).get("description", "")
        dataset_records.append({"caption": description, "media_path": video_path})

        downloaded += 1
        print(f"[{downloaded}/{args.max_videos}] {video_filename} "
              f"({meta.get('duration_seconds', '?')}s) - {meta.get('content_fine_category', '?')}")

        if downloaded >= args.max_videos:
            break

    _save_dataset_json(output_dir, dataset_records, downloaded)


def _save_dataset_json(output_dir: str, records: list[dict], count: int) -> None:
    path = os.path.join(output_dir, "dataset.json")
    with open(path, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Downloaded {count} videos.")
    print(f"Dataset JSON: {path}")
    print(f"\nNext — preprocess for training:")
    print(f"  uv run python packages/ltx-trainer/scripts/process_dataset.py \\")
    print(f"      {path} \\")
    print(f"      --resolution-buckets '768x448x49' \\")
    print(f"      --model-path /nfs/hanpeng/huggingface/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \\")
    print(f"      --text-encoder-path /nfs/hanpeng/huggingface/models/gemma-3-12b-it-qat-q4_0-unquantized")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output_dir = args.output_dir or default_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    if args.build_index:
        build_index(output_dir)
        return

    if args.list_categories:
        list_categories(output_dir)
        return

    print(f"Output directory : {output_dir}")
    print(f"Category         : {args.category or '(all)'}")
    print(f"Fine category    : {args.fine_category or '(all)'}")
    print(f"Keyword          : {args.keyword or '(none)'}")
    print(f"Duration range   : {args.min_duration}s – {args.max_duration}s")
    print(f"Max videos       : {args.max_videos}\n")

    if args.no_index:
        download_via_streaming(args, output_dir)
    else:
        idx = index_path(output_dir)
        if not os.path.exists(idx):
            print(f"No local index found at {idx}.")
            print("Tip: run --build-index first for much faster filtering.")
            print("Falling back to streaming mode...\n")
            download_via_streaming(args, output_dir)
        else:
            download_via_index(args, output_dir)


if __name__ == "__main__":
    main()
