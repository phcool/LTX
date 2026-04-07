"""
Download videos from Pexels API for LTX-2 training.

Usage:
    # Single query
    python download_pexels.py --query "nature forest" --max-videos 200

    # Multiple queries
    python download_pexels.py --queries "nature" "ocean waves" "mountain landscape" --max-videos 100

    # With quality and duration filters
    python download_pexels.py --query "cinematic city" --max-videos 100 \
        --min-width 1920 --min-duration 5 --max-duration 60

    # Specify output directory (defaults to $HF_HOME/datasets/pexels)
    python download_pexels.py --query "nature" --output-dir /nfs/hanpeng/data/pexels

API key: set PEXELS_API_KEY env var, or pass --api-key.
Get a free key at https://www.pexels.com/api/
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path

import requests


PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"
DEFAULT_PER_PAGE = 80  # max allowed by Pexels API
DEFAULT_OUTPUT_DIR = "/nfs/hanpeng/huggingface/datasets/pexels"
VIDEO_FILENAME_RE = re.compile(r"video_(\d+)\.(?:mp4|json)$")
ANIMAL_LONGTAIL_QUERIES = [
    "animal rescue",
    "animal shelter",
    "animal sanctuary",
    "pet grooming",
    "veterinarian animal",
    "dog running",
    "dog playing",
    "dog park",
    "dog beach",
    "puppy playing",
    "cat playing",
    "cat sleeping",
    "kitten playing",
    "bird flying",
    "bird feeding",
    "bird nest",
    "parrot talking",
    "owl forest",
    "eagle flying",
    "horse running",
    "horse riding",
    "horse farm",
    "cow grazing",
    "goat farm",
    "sheep field",
    "pig farm",
    "rabbit eating",
    "hamster pet",
    "deer forest",
    "fox wild",
    "wolf pack",
    "bear forest",
    "lion resting",
    "tiger walking",
    "leopard wild",
    "cheetah running",
    "elephant walking",
    "giraffe safari",
    "zebra herd",
    "monkey climbing",
    "gorilla forest",
    "chimpanzee wild",
    "panda eating",
    "koala tree",
    "kangaroo jumping",
    "dolphin swimming",
    "whale ocean",
    "seal beach",
    "sea lion coast",
    "otter river",
    "fish underwater",
    "shark swimming",
    "turtle swimming",
    "frog pond",
    "snake crawling",
    "lizard reptile",
    "bee pollinating",
    "butterfly flower",
    "dragonfly pond",
    "zoo animals",
    "farm animals",
    "wild animals",
    "underwater animals",
]
QUERY_PRESETS = {
    "animals_longtail": ANIMAL_LONGTAIL_QUERIES,
}


def get_default_download_workers() -> int:
    cpu_threads = os.cpu_count() or 8
    return min(16, max(4, cpu_threads // 16))


def load_dotenv(env_path: str | None = None) -> None:
    """Load key=value pairs from a .env file into os.environ (no third-party deps)."""
    candidates = [env_path] if env_path else [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key, value)
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Pexels videos for LTX-2 training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--api-key", type=str, default=None,
                        help="Pexels API key. Falls back to PEXELS_API_KEY in .env or env var.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Output directory. Defaults to {DEFAULT_OUTPUT_DIR}.")
    parser.add_argument("--env-file", type=str, default=None,
                        help="Path to .env file (default: .env next to this script).")

    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query", type=str,
                             help="Single search query (e.g. 'nature forest')")
    query_group.add_argument("--queries", type=str, nargs="+",
                             help="Multiple search queries; videos are pooled together")
    query_group.add_argument("--queries-file", type=str,
                             help="Path to a text file with one query per line.")
    query_group.add_argument("--query-preset", type=str, choices=sorted(QUERY_PRESETS),
                             help="Use a built-in query preset.")

    parser.add_argument("--max-videos", type=int, default=200,
                        help="Total maximum videos to download across all queries (default: 200)")
    parser.add_argument("--min-duration", type=float, default=5.0,
                        help="Minimum video duration in seconds (default: 5.0)")
    parser.add_argument("--max-duration", type=float, default=60.0,
                        help="Maximum video duration in seconds (default: 60.0)")
    parser.add_argument("--min-width", type=int, default=1280,
                        help="Minimum video width in pixels (default: 1280)")
    parser.add_argument("--prefer-quality", type=str, default="hd",
                        choices=["hd", "sd"],
                        help="Preferred video quality tier (default: hd)")
    parser.add_argument("--locale", type=str, default="en-US",
                        help="Search locale (default: en-US)")
    parser.add_argument("--orientation", type=str, default=None,
                        choices=["landscape", "portrait", "square"],
                        help="Filter by video orientation")
    parser.add_argument("--size", type=str, default=None,
                        choices=["large", "medium", "small"],
                        help="Filter by video size")
    parser.add_argument("--rate-limit-delay", type=float, default=0.25,
                        help="Seconds to wait between API requests (default: 0.25)")
    parser.add_argument("--download-workers", type=int, default=None,
                        help="Concurrent download workers. Defaults to an auto value based on CPU threads.")
    parser.add_argument("--api-retries", type=int, default=6,
                        help="Max retries for API search requests on transient failures like HTTP 429 (default: 6)")
    parser.add_argument("--api-backoff-base", type=float, default=5.0,
                        help="Initial backoff in seconds for API retries (default: 5.0)")
    parser.add_argument("--api-backoff-max", type=float, default=120.0,
                        help="Maximum backoff in seconds for API retries (default: 120.0)")
    return parser.parse_args()


def load_queries_from_file(path: str) -> list[str]:
    queries: list[str] = []
    with open(path) as f:
        for line in f:
            query = line.strip()
            if not query or query.startswith("#"):
                continue
            queries.append(query)
    return queries


def dedupe_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for query in queries:
        normalized = query.strip()
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def get_api_key(args: argparse.Namespace) -> str:
    key = args.api_key or os.environ.get("PEXELS_API_KEY", "")
    if not key:
        print("ERROR: No Pexels API key found.")
        print("  Add to .env:  PEXELS_API_KEY=your_key")
        print("  Set env var:  export PEXELS_API_KEY=your_key")
        print("  Or pass:      --api-key your_key")
        print("  Get a free key at https://www.pexels.com/api/")
        sys.exit(1)
    return key


def pick_best_video_file(video_files: list[dict], prefer_quality: str, min_width: int) -> dict | None:
    """Pick the best mp4 file: prefer `prefer_quality` tier, highest resolution >= min_width."""
    mp4_files = [
        vf for vf in video_files
        if vf.get("file_type") == "video/mp4"
        and vf.get("width") is not None
        and vf.get("width", 0) >= min_width
        and vf.get("quality") in ("hd", "sd")
    ]
    if not mp4_files:
        return None

    preferred = [vf for vf in mp4_files if vf.get("quality") == prefer_quality]
    pool = preferred if preferred else mp4_files
    return max(pool, key=lambda vf: vf.get("width", 0))


def search_page(
    session,
    api_key: str,
    query: str,
    page: int,
    per_page: int,
    locale: str,
    orientation: str | None,
    size: str | None,
) -> dict:
    params: dict = {
        "query": query,
        "page": page,
        "per_page": per_page,
        "locale": locale,
    }
    if orientation:
        params["orientation"] = orientation
    if size:
        params["size"] = size

    resp = session.get(
        PEXELS_SEARCH_URL,
        headers={"Authorization": api_key},
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def search_page_with_retry(
    session,
    api_key: str,
    query: str,
    page: int,
    per_page: int,
    locale: str,
    orientation: str | None,
    size: str | None,
    max_retries: int,
    backoff_base: float,
    backoff_max: float,
) -> dict:
    attempt = 0
    while True:
        try:
            return search_page(
                session=session,
                api_key=api_key,
                query=query,
                page=page,
                per_page=per_page,
                locale=locale,
                orientation=orientation,
                size=size,
            )
        except requests.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code != 429 or attempt >= max_retries:
                raise

            retry_after = None
            if e.response is not None:
                retry_after_header = e.response.headers.get("Retry-After")
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except ValueError:
                        retry_after = None

            delay = retry_after if retry_after is not None else min(backoff_base * (2 ** attempt), backoff_max)
            attempt += 1
            print(
                f"HTTP 429 for query={query!r} page={page}. "
                f"Sleeping {delay:.1f}s before retry {attempt}/{max_retries}...",
                flush=True,
            )
            time.sleep(delay)
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt >= max_retries:
                raise
            delay = min(backoff_base * (2 ** attempt), backoff_max)
            attempt += 1
            print(
                f"Transient API error for query={query!r} page={page}: {e}. "
                f"Sleeping {delay:.1f}s before retry {attempt}/{max_retries}...",
                flush=True,
            )
            time.sleep(delay)


def download_video(session, url: str, dest_path: str) -> bool:
    """Stream-download a video file to dest_path. Returns True on success."""
    try:
        with session.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"    Download failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def parse_video_index(path_like: str | os.PathLike[str]) -> int | None:
    match = VIDEO_FILENAME_RE.match(Path(path_like).name)
    if not match:
        return None
    return int(match.group(1))


def load_existing_pexels_ids(metadata_dir: str) -> set[int]:
    existing_ids: set[int] = set()
    metadata_path = Path(metadata_dir)
    if not metadata_path.exists():
        return existing_ids

    for meta_file in metadata_path.glob("*.json"):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            pexels_id = meta.get("pexels_id")
            if pexels_id is not None:
                existing_ids.add(int(pexels_id))
        except Exception:
            continue

    return existing_ids


def get_next_video_index(videos_dir: str, metadata_dir: str, dataset_records: list[dict]) -> int:
    max_index = -1

    for directory in (videos_dir, metadata_dir):
        directory_path = Path(directory)
        if not directory_path.exists():
            continue
        for path in directory_path.iterdir():
            idx = parse_video_index(path.name)
            if idx is not None:
                max_index = max(max_index, idx)

    for record in dataset_records:
        media_path = record.get("media_path")
        if media_path:
            idx = parse_video_index(media_path)
            if idx is not None:
                max_index = max(max_index, idx)

    return max_index + 1


def download_video_worker(url: str, dest_path: str) -> tuple[bool, str | None]:
    try:
        import requests
    except ImportError:
        return False, "requests is not installed"

    try:
        with requests.Session() as session:
            ok = download_video(session, url, dest_path)
        return ok, None if ok else "download failed"
    except Exception as e:
        return False, str(e)


def fetch_and_download(
    args: argparse.Namespace,
    api_key: str,
    queries: list[str],
    output_dir: str,
) -> None:
    videos_dir = os.path.join(output_dir, "videos")
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    dataset_json_path = os.path.join(output_dir, "dataset.json")
    # Resume: load already-downloaded records
    if os.path.exists(dataset_json_path):
        with open(dataset_json_path) as f:
            dataset_records: list[dict] = json.load(f)
        print(f"Resuming: {len(dataset_records)} videos already downloaded.")
    else:
        dataset_records = []

    existing_pexels_ids = load_existing_pexels_ids(metadata_dir)
    if existing_pexels_ids:
        print(f"Loaded {len(existing_pexels_ids)} existing Pexels IDs for deduplication.")

    already_downloaded = len(dataset_records)
    downloaded = 0
    skipped_duplicates = 0
    budget = args.max_videos - already_downloaded
    if budget <= 0:
        print(f"Already have {already_downloaded} videos (target: {args.max_videos}). Nothing to do.")
        return

    session = requests.Session()
    download_workers = args.download_workers or get_default_download_workers()
    next_video_index = get_next_video_index(videos_dir, metadata_dir, dataset_records)

    print(f"Download workers : {download_workers}")

    for query in queries:
        if downloaded >= budget:
            break

        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        page = 1
        query_downloaded = 0

        while downloaded < budget:
            print(f"  Page {page} ...", end=" ", flush=True)
            try:
                data = search_page_with_retry(
                    session=session,
                    api_key=api_key,
                    query=query,
                    page=page,
                    per_page=min(DEFAULT_PER_PAGE, budget - downloaded + 10),
                    locale=args.locale,
                    orientation=args.orientation,
                    size=args.size,
                    max_retries=args.api_retries,
                    backoff_base=args.api_backoff_base,
                    backoff_max=args.api_backoff_max,
                )
            except Exception as e:
                print(f"API error: {e}")
                break

            total_results = data.get("total_results", 0)
            videos = data.get("videos", [])
            print(f"{len(videos)} results (total available: {total_results})")

            if not videos:
                break

            page_candidates: list[dict] = []
            for video in videos:
                if downloaded >= budget:
                    break

                duration = video.get("duration", 0)
                if not (args.min_duration <= duration <= args.max_duration):
                    continue

                best_file = pick_best_video_file(
                    video.get("video_files", []),
                    args.prefer_quality,
                    args.min_width,
                )
                if not best_file:
                    continue

                video_id = video["id"]
                if int(video_id) in existing_pexels_ids:
                    skipped_duplicates += 1
                    continue

                existing_pexels_ids.add(int(video_id))

                global_idx = next_video_index
                next_video_index += 1
                video_filename = f"video_{global_idx:05d}.mp4"
                video_path = os.path.join(videos_dir, video_filename)
                page_candidates.append({
                    "video_id": video_id,
                    "global_idx": global_idx,
                    "duration": duration,
                    "best_file": best_file,
                    "video": video,
                    "video_path": video_path,
                })

                if downloaded + len(page_candidates) >= budget:
                    break

            if page_candidates:
                with concurrent.futures.ThreadPoolExecutor(max_workers=download_workers) as executor:
                    futures: dict[concurrent.futures.Future, dict] = {}
                    for candidate in page_candidates:
                        display_idx = already_downloaded + downloaded + len(futures) + 1
                        best_file = candidate["best_file"]
                        print(
                            f"  [{display_idx}/{args.max_videos}] "
                            f"id={candidate['video_id']} {best_file['width']}x{best_file['height']} "
                            f"{candidate['duration']}s ({best_file['quality']}) ... queued",
                            flush=True,
                        )
                        futures[executor.submit(download_video_worker, best_file["link"], candidate["video_path"])] = candidate

                    for future in concurrent.futures.as_completed(futures):
                        candidate = futures[future]
                        ok, error = future.result()
                        if not ok:
                            print(f"    Download failed for id={candidate['video_id']}: {error}")
                            existing_pexels_ids.discard(int(candidate["video_id"]))
                            continue

                        video = candidate["video"]
                        best_file = candidate["best_file"]
                        duration = candidate["duration"]
                        video_path = candidate["video_path"]
                        global_idx = candidate["global_idx"]

                        user_name = video.get("user", {}).get("name", "")
                        pexels_url = video.get("url", "")
                        caption = (
                            f"A {duration}-second video clip. "
                            f"Source: {pexels_url}. Filmed by {user_name}."
                        )

                        meta = {
                            "pexels_id": candidate["video_id"],
                            "pexels_url": pexels_url,
                            "query": query,
                            "duration_seconds": duration,
                            "width": video.get("width"),
                            "height": video.get("height"),
                            "photographer": user_name,
                            "selected_file": {
                                "quality": best_file["quality"],
                                "width": best_file["width"],
                                "height": best_file["height"],
                                "link": best_file["link"],
                            },
                        }

                        meta_path = os.path.join(metadata_dir, f"video_{global_idx:05d}.json")
                        with open(meta_path, "w") as f:
                            json.dump(meta, f, ensure_ascii=False, indent=2)

                        dataset_records.append({
                            "caption": caption,
                            "media_path": video_path,
                        })
                        downloaded += 1
                        query_downloaded += 1

                        with open(dataset_json_path, "w") as f:
                            json.dump(dataset_records, f, ensure_ascii=False, indent=2)

                        print(f"    OK id={candidate['video_id']} -> video_{global_idx:05d}.mp4")

            # Check if more pages exist
            next_page_start = page * DEFAULT_PER_PAGE + 1
            if next_page_start > total_results or not videos:
                print(f"  No more pages for query '{query}'.")
                break

            page += 1
            time.sleep(args.rate_limit_delay)

        print(f"Query '{query}': downloaded {query_downloaded} videos.")

    total = already_downloaded + downloaded
    print(f"\n{'='*60}")
    print(f"Done. Total videos: {total} (this run: {downloaded})")
    print(f"Skipped duplicate Pexels IDs: {skipped_duplicates}")
    print(f"Dataset JSON: {dataset_json_path}")
    print(f"\nNext — preprocess for training:")
    print(f"  uv run python packages/ltx-trainer/scripts/process_dataset.py \\")
    print(f"      {dataset_json_path} \\")
    print(f"      --resolution-buckets '768x448x49' \\")
    print(f"      --model-path /nfs/hanpeng/huggingface/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \\")
    print(f"      --text-encoder-path /nfs/hanpeng/huggingface/models/gemma-3-12b-it-qat-q4_0-unquantized")


def main() -> None:
    args = parse_args()
    load_dotenv(getattr(args, "env_file", None))
    api_key = get_api_key(args)
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    if args.queries:
        queries = args.queries
    elif args.query:
        queries = [args.query]
    elif args.queries_file:
        queries = load_queries_from_file(args.queries_file)
    else:
        queries = QUERY_PRESETS[args.query_preset]

    queries = dedupe_queries(queries)
    if not queries:
        print("ERROR: No queries resolved after loading and deduplication.")
        sys.exit(1)

    print(f"Output directory : {output_dir}")
    print(f"Queries          : {len(queries)} query(s)")
    print(f"Max videos       : {args.max_videos}")
    print(f"Duration range   : {args.min_duration}s – {args.max_duration}s")
    print(f"Min width        : {args.min_width}px")
    print(f"Prefer quality   : {args.prefer_quality}")
    print(f"Download workers : {args.download_workers or get_default_download_workers()} (auto if unspecified)")

    fetch_and_download(args, api_key, queries, output_dir)


if __name__ == "__main__":
    main()
