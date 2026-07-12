import pandas as pd
import re
import os
from collections import Counter
from pathlib import Path

# --- Paths (relative to the repo, so this runs on ANY machine) ---
# This file lives at <repo>/coding/code/data_preocessing.py, so the repo root
# is two folders up (parents[0]=code, parents[1]=coding, parents[2]=repo root).
# Building paths from __file__ means we never hardcode C:/Users/Admin/... again.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"                       # input:  data/<year>/*.srt
OUTPUT_CSV = REPO_ROOT / "coding" / "data_csv.csv"  # output: the central artifact

# Define the list of common words to exclude
common_exclusions = {'-','♪','i', 'you', 'to', 'the', 'a', 'and', 'it', 'is', 'that', 'of','s', 't', 'what', 'in', 'me', 'this', 'on', 'sir', 'get','for', 'she', 'be', 'eve', 'not', 'have', 'all', 'her', 'was', 'my','can', 'oh', 'no', 'we', 'well', 'annie', 'be', 'he', 'like', 'don'}

def read_text_with_fallback(file_path):
    """Read a subtitle file as text, trying common encodings in order.

    Old .srt files are often Windows-1252 (cp1252), not UTF-8. latin-1 maps
    all 256 byte values so it never raises -- it is the guaranteed last resort.
    This means one oddly-encoded file can no longer crash the whole run.
    """
    for encoding in ('utf-8', 'cp1252', 'latin-1'):
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    # Unreachable in practice (latin-1 above cannot fail), but explicit and safe:
    with open(file_path, 'r', encoding='latin-1', errors='replace') as file:
        return file.read()

def parse_srt_excluding_common(file_path):
    content = read_text_with_fallback(file_path)

    # Extract all timestamps
    timestamps = re.findall(r'\d{2}:\d{2}:\d{2},\d{3}', content)
    if timestamps:
        last_timestamp = timestamps[-1]
        hours, minutes, seconds_milliseconds = last_timestamp.split(':')
        seconds, milliseconds = seconds_milliseconds.split(',')
        total_minutes = int(hours) * 60 + int(minutes) + int(seconds) / 60 + int(milliseconds) / (60 * 1000)
    else:
        total_minutes = 0

    # Remove timestamps and numbers
    lines = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', content)
    lines = re.sub(r'\d+', '', lines)
    # Replace newlines with a SPACE, not nothing -- otherwise the last word of
    # one subtitle block glues onto the first word of the next ("world" + "This"
    # -> "worldThis"), corrupting word counts and the text fed to the models.
    lines = lines.replace('\n', ' ')
    lines = re.sub(r'\s+', ' ', lines).strip()
    lines = lines.replace('-', '')
    lines = lines.replace('♪', '')
    lines = re.sub(r'</?i>', '', lines)
    lines = re.sub(r'</?b>', '', lines)

    # Extract words
    words = re.findall(r'\b\w+\b', lines.lower())
    word_count = len(words)

    # Filter out common exclusions
    filtered_words = [word for word in words if word not in common_exclusions]

    # Get the most common words
    common_words = Counter(filtered_words).most_common(10)
    top_ten_words = [word for word, _ in common_words]

    return word_count, total_minutes, top_ten_words,lines

def process_srt_files_excluding_common(directory):
    data = []
    skipped = []  # (filename, reason) -- so we never silently drop a movie
    for year_folder in sorted(os.listdir(directory)):
        year_path = os.path.join(directory, year_folder)
        if not os.path.isdir(year_path):
            continue
        for srt_file in sorted(os.listdir(year_path)):
            if not srt_file.endswith('.srt'):
                continue
            file_path = os.path.join(year_path, srt_file)
            # Strip the .srt extension BEFORE cleaning, so "Eve.srt" -> "Eve",
            # not "Evesrt". Then drop punctuation/spaces to keep the dataset's
            # existing movie-name convention (e.g. "AllAboutEve").
            movie_name = re.sub(r'[^\w\s]', '', Path(srt_file).stem).replace(' ', '')
            try:
                word_count, total_minutes, top_ten_words, content = parse_srt_excluding_common(file_path)
            except Exception as exc:  # one bad file must not kill the whole run
                skipped.append((srt_file, f"parse error: {exc}"))
                continue
            if word_count == 0:  # e.g. the 3 empty 0-byte files
                skipped.append((srt_file, "empty / no dialogue"))
                continue
            data.append({
                'movie': movie_name,
                'year': year_folder,
                'numberofwords': word_count,
                'time': f"{total_minutes:.2f}mins",
                'toptenwords': top_ten_words,
                'bodyContent': content
            })
    if skipped:
        print(f"Skipped {len(skipped)} file(s):")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")
    return data

def main():
    # Process the SRT files and gather the data
    srt_data_excluding_common = process_srt_files_excluding_common(DATA_DIR)

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(srt_data_excluding_common)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} movies to {OUTPUT_CSV}")


# This guard means: run the pipeline only when this file is executed directly
# (`python data_preocessing.py`). When another file *imports* it (like our tests),
# nothing runs automatically -- we just get access to the functions above.
if __name__ == "__main__":
    main()


