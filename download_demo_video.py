import os
import requests
from pathlib import Path

DEST_DIR = Path("demo_media")
DEST_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FILE = DEST_DIR / "highway_collision_demo.webm"
METADATA_FILE = DEST_DIR / "SOURCE_METADATA.txt"

SOURCE_URL = "https://upload.wikimedia.org/wikipedia/commons/b/b1/Multiple_cars_rear-end_collision_on_highway.webm"
SOURCE_PAGE = "https://commons.wikimedia.org/wiki/File:Multiple_cars_rear-end_collision_on_highway.webm"
LICENSE = "Public Domain (Automated highway CCTV monitoring recording without human creative input, uploaded via video2commons)"

print(f"Downloading demonstration video from: {SOURCE_URL}...")
headers = {
    "User-Agent": "SafeTransit-AI-ResearchBot/1.0 (https://github.com/Shameer28tm/safetransit-ai; contact: student@hackathon.org)"
}

try:
    with requests.get(SOURCE_URL, headers=headers, stream=True, timeout=20) as r:
        r.raise_for_status()
        total_len = int(r.headers.get('content-length', 0))
        print(f"Content length: {total_len / (1024*1024):.2f} MB")
        downloaded = 0
        with open(TARGET_FILE, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    print(f"Downloaded: {downloaded / (1024*1024):.2f} MB / {total_len / (1024*1024):.2f} MB", end="\r")
    print(f"\nDownload completed: {TARGET_FILE} ({os.path.getsize(TARGET_FILE)} bytes)")

    # Record metadata
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        f.write(f"Source URL: {SOURCE_URL}\n")
        f.write(f"Source Page: {SOURCE_PAGE}\n")
        f.write(f"File Name: {TARGET_FILE.name}\n")
        f.write(f"License / Rights: {LICENSE}\n")
        f.write(f"Description: Multi-vehicle highway rear-end collision recorded by highway surveillance.\n")
    print(f"Saved provenance record to {METADATA_FILE}")

except Exception as e:
    print(f"Error downloading video: {e}")
