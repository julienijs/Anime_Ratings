import requests
import time
import csv
import os

BASE_URL = "https://api.jikan.moe/v4/anime"
REQUEST_DELAY = 0.5
MAX_RETRIES = 5

# --------------------------------------------------
# Helper: GET with retry + exponential backoff
# --------------------------------------------------
def get_with_retry(url, params, max_retries=MAX_RETRIES):
    for attempt in range(1, max_retries + 1):
        response = requests.get(url, params=params)

        if response.status_code == 200:
            return response

        if response.status_code == 429:
            wait = REQUEST_DELAY * (2 ** attempt)
            print(f"Rate limited (429). Sleeping {wait:.1f}s and retrying...")
            time.sleep(wait)
            continue

        response.raise_for_status()

    raise RuntimeError("Max retries exceeded due to repeated rate limiting")

# --------------------------------------------------
# Main logic
# --------------------------------------------------
def main():
    start_time = time.time()
    all_anime = []

    params = {
        "type": "tv",
        "page": 1,
        "limit": 25
    }

    # ---- Initial request ----
    response = get_with_retry(BASE_URL, params)
    data = response.json()

    total_pages = data["pagination"]["last_visible_page"]
    print(f"Total pages to fetch: {total_pages}")

    # ---- Page processor ----
    def process_page(page_data):
        for anime in page_data:
            all_anime.append({
                "title": anime["title"],
                "year": anime["aired"]["prop"]["from"]["year"],
                "score": anime["score"],
                "scored_by": anime["scored_by"],
                "members": anime["members"],   # <-- WATCHERS / POPULARITY
                "rank": anime["rank"],
                "genres": ", ".join(g["name"] for g in anime["genres"]),
                "demographics": ", ".join(d["name"] for d in anime["demographics"]),
            })

    # ---- Process page 1 ----
    process_page(data["data"])

    # ---- Remaining pages ----
    for page in range(2, total_pages + 1):
        time.sleep(REQUEST_DELAY)

        params["page"] = page
        response = get_with_retry(BASE_URL, params)
        data = response.json()

        process_page(data["data"])

        # ---- ETA ----
        elapsed = time.time() - start_time
        avg_time_per_page = elapsed / page
        remaining_pages = total_pages - page
        eta_minutes = (avg_time_per_page * remaining_pages) / 60

        print(
            f"Page {page}/{total_pages} | "
            f"Elapsed: {elapsed/60:.1f} min | "
            f"ETA: {eta_minutes:.1f} min"
        )

    # ---- Save to CSV ----
    output_dir = "Anime_Data"
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "tv_anime_ratings.csv")

    fieldnames = [
        "title",
        "year",
        "score",
        "scored_by",
        "members",
        "rank",
        "genres",
        "demographics"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_anime)

    total_time = time.time() - start_time
    print(f"\nFinished in {total_time/60:.1f} minutes")
    print(f"Collected {len(all_anime)} TV anime")
    print(f"Saved to {csv_path}")

# --------------------------------------------------
if __name__ == "__main__":
    main()
