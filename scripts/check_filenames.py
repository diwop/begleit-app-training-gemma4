#!/usr/bin/env python3
"""
Script to validate and auto-fix filename patterns in data/raw.
- Deletes .docx files
- Replaces space with underscore in 'Leichte Sprache' -> 'Leichte_Sprache'
- Strips trailing spaces before .txt extension
- Fixes 3-digit prefixes (e.g. 115_ -> 0115_)
- Reports remaining invalid files matching NNNN_Standardsprache.txt or NNNN_Leichte_Sprache.txt
"""

import sys
import re
from pathlib import Path

# Strict expected pattern: 4 digits NNNN, underscore _, followed by 'Standardsprache' or 'Leichte_Sprache', ending in .txt
PATTERN = re.compile(r"^\d{4}_(Standardsprache|Leichte_Sprache)\.txt$")

# Files/extensions to ignore during validation
IGNORE_PREFIXES = (".",)
IGNORE_SUFFIXES = (".dvc",)
IGNORE_FILES = (".gitkeep", ".gitignore")

def fix_and_check_filenames(directory: Path) -> list[tuple[str, str]]:
    """Auto-fix space in Leichte_Sprache, trailing spaces & docx files, then return remaining mismatched files."""
    if not directory.exists():
        print(f"[ERROR] Directory '{directory}' does not exist.")
        return []

    print("[INFO] Applying automatic fixes (unifying 'Leichte_Sprache', deleting .docx files, fixing 3-digit prefixes)...")
    deleted_count = 0
    renamed_count = 0

    for path in sorted(directory.iterdir()):
        filename = path.name

        if filename in IGNORE_FILES or filename.startswith(IGNORE_PREFIXES) or filename.endswith(IGNORE_SUFFIXES):
            continue

        # 1. Delete .docx files
        if filename.endswith(".docx"):
            print(f"  [DELETED] '{filename}'")
            path.unlink()
            deleted_count += 1
            continue

        cleaned_name = filename

        # 2. Fix trailing spaces before .txt
        if re.search(r"\s+\.txt$", cleaned_name):
            cleaned_name = re.sub(r"\s+\.txt$", ".txt", cleaned_name)

        # 3. Unify 'Leichte Sprache' -> 'Leichte_Sprache'
        if "Leichte Sprache" in cleaned_name:
            cleaned_name = cleaned_name.replace("Leichte Sprache", "Leichte_Sprache")

        # 4. Fix 3-digit prefixes (e.g. 115_ -> 0115_)
        if re.match(r"^\d{3}_", cleaned_name):
            cleaned_name = "0" + cleaned_name

        # Collapse internal multiple spaces if any remain
        cleaned_name = re.sub(r"\s{2,}", " ", cleaned_name)

        new_path = path.parent / cleaned_name
        if new_path != path:
            if new_path.exists():
                print(f"  [DELETED DUPLICATE] '{filename}' (identical file '{cleaned_name}' already exists)")
                path.unlink()
                deleted_count += 1
            else:
                print(f"  [RENAMED] '{filename}' -> '{cleaned_name}'")
                path.rename(new_path)
                renamed_count += 1

    print(f"[INFO] Auto-fix complete: Deleted {deleted_count} file(s), Renamed {renamed_count} file(s).\n")

    # Pass 2: Inspect remaining files
    mismatches = []
    for path in sorted(directory.iterdir()):
        filename = path.name

        if filename in IGNORE_FILES or filename.startswith(IGNORE_PREFIXES) or filename.endswith(IGNORE_SUFFIXES):
            continue

        if not PATTERN.match(filename):
            reasons = []
            if not filename.endswith(".txt"):
                reasons.append(f"non-.txt extension ({path.suffix})")
            if "Standartsprache" in filename:
                reasons.append("typo 'Standartsprache' (expected 'Standardsprache')")
            if not re.match(r"^\d{4}_", filename):
                reasons.append("invalid prefix (expected 4 digits NNNN_)")
            if "Leichte Sprache" in filename:
                reasons.append("space in 'Leichte Sprache' (expected 'Leichte_Sprache')")

            reason_str = ", ".join(reasons) if reasons else "invalid pattern"
            mismatches.append((filename, reason_str))

    return mismatches

def main() -> None:
    dir_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    print(f"Checking and fixing filenames in '{dir_path}'...\n")

    mismatches = fix_and_check_filenames(dir_path)

    if not mismatches:
        print(" [SUCCESS] All filenames in data/raw match the expected pattern!")
    else:
        print(f" [WARNING] Found {len(mismatches)} remaining mismatched file(s):\n")
        for idx, (fname, reason) in enumerate(mismatches, 1):
            print(f"  {idx:3d}. '{fname}' -> [{reason}]")
        sys.exit(1)

if __name__ == "__main__":
    main()
