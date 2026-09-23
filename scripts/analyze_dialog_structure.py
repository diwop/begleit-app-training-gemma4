#!/usr/bin/env python3
"""
analyze_dialog_structure.py

Analyzes dialog files in a target directory (default: data/raw_dialogs/) to verify
they follow the regular turn structure:
    [<user>], <partner>, <partner simple translation|formatting>, <user>, ...

Expected Dialogue Grammar:
--------------------------
1. Start: May optionally start with <user>, or start directly with <partner>.
2. Cycles:
   - <partner> MUST be immediately followed by <partner simple translation> or <partner simple formatting>.
   - <partner simple translation|formatting> MUST be followed by <user> (or be the final turn in the file).
   - <user> MUST be followed by <partner> (or be the final turn in the file).
3. End:
   - May end after a <user> turn, or after a <partner simple translation|formatting> turn.
   - Must NEVER end with an un-translated <partner> turn.
4. XML / Tag Syntax:
   - Recognized opening tags: <user>, <partner>, <partner simple translation>, <partner simple formatting>
   - Proper tag pairing: <user> must be closed by </user>. <partner> and its translation variants closed by </partner>.
   - No untagged text outside tags, and no empty turn tags.
5. Duplication Check:
   - Detects duplicated identical partner or user turns (e.g., untranslated turn pasted twice).

Usage:
------
    python scripts/analyze_dialog_structure.py
    python scripts/analyze_dialog_structure.py --dir data/raw_dialogs
    python scripts/analyze_dialog_structure.py --verbose
    python scripts/analyze_dialog_structure.py --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

# Allowed tags in raw dialog files
VALID_OPEN_TAGS = {
    "user",
    "partner",
    "partner simple translation",
    "partner simple formatting",
}

TAG_REGEX = re.compile(r"<(/?)([a-zA-Z0-9_ ]+)>")


class Issue:
    def __init__(
        self,
        category: str,
        line: int,
        message: str,
        snippet: str | None = None,
    ) -> None:
        self.category = category
        self.line = line
        self.message = message
        self.snippet = snippet

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "category": self.category,
            "line": self.line,
            "message": self.message,
        }
        if self.snippet:
            d["snippet"] = self.snippet
        return d


class DialogAnalysis:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.issues: list[Issue] = []
        self.turn_counts: dict[str, int] = {
            "user": 0,
            "partner": 0,
            "partner_translation": 0,
        }
        self.total_turns = 0

    @property
    def is_regular(self) -> bool:
        return len(self.issues) == 0


def analyze_file(filepath: Path) -> DialogAnalysis:
    analysis = DialogAnalysis(filepath)
    content = filepath.read_text(encoding="utf-8")
    lines = content.splitlines()

    # 1. Parse tags with line numbers
    tag_events: list[tuple[int, bool, str, str]] = []  # (line_no, is_closing, tag_name, raw_tag)
    for line_idx, line in enumerate(lines, 1):
        for m in TAG_REGEX.finditer(line):
            is_closing = bool(m.group(1))
            tag_name = m.group(2).strip()
            raw_tag = m.group(0)
            tag_events.append((line_idx, is_closing, tag_name, raw_tag))

    # 2. Check for unrecognized tags
    for line_no, is_closing, tag_name, raw_tag in tag_events:
        if is_closing:
            if tag_name not in ("user", "partner", "partner simple translation", "partner simple formatting"):
                analysis.issues.append(
                    Issue(
                        category="Syntax: Unrecognized Tag",
                        line=line_no,
                        message=f"Unrecognized closing tag '{raw_tag}'",
                    )
                )
        else:
            if tag_name not in VALID_OPEN_TAGS:
                analysis.issues.append(
                    Issue(
                        category="Syntax: Unrecognized Tag",
                        line=line_no,
                        message=f"Unrecognized opening tag '{raw_tag}'",
                    )
                )

    # 3. Check tag nesting and proper closure
    open_stack: list[tuple[int, str]] = []  # (line_no, tag_name)
    for line_no, is_closing, tag_name, raw_tag in tag_events:
        if not is_closing:
            open_stack.append((line_no, tag_name))
        else:
            if not open_stack:
                analysis.issues.append(
                    Issue(
                        category="Syntax: Mismatched Closing Tag",
                        line=line_no,
                        message=f"Closing tag '</{tag_name}>' found without corresponding opening tag",
                    )
                )
            else:
                open_line, open_tag = open_stack.pop()
                if open_tag == "user" and tag_name != "user":
                    analysis.issues.append(
                        Issue(
                            category="Syntax: Mismatched Closing Tag",
                            line=line_no,
                            message=f"<user> (opened at line {open_line}) was closed by '</{tag_name}>' instead of '</user>'",
                            snippet=lines[line_no - 1].strip() if line_no <= len(lines) else None,
                        )
                    )
                elif open_tag in ("partner", "partner simple translation", "partner simple formatting") and tag_name not in (
                    "partner",
                    open_tag,
                ):
                    analysis.issues.append(
                        Issue(
                            category="Syntax: Mismatched Closing Tag",
                            line=line_no,
                            message=f"<{open_tag}> (opened at line {open_line}) was closed by '</{tag_name}>'",
                            snippet=lines[line_no - 1].strip() if line_no <= len(lines) else None,
                        )
                    )

    for open_line, open_tag in open_stack:
        analysis.issues.append(
            Issue(
                category="Syntax: Unclosed Tag",
                line=open_line,
                message=f"Opening tag '<{open_tag}>' at line {open_line} is never closed",
                snippet=lines[open_line - 1].strip() if open_line <= len(lines) else None,
            )
        )

    # 4. Check for untagged text outside of tags
    # Strip recognized blocks and see if any non-whitespace characters remain
    cleaned_content = re.sub(
        r"<(user|partner|partner simple translation|partner simple formatting)>.*?</(user|partner|partner simple translation|partner simple formatting)>",
        "",
        content,
        flags=re.DOTALL,
    )
    unparsed_text = cleaned_content.strip()
    if unparsed_text:
        snippet = unparsed_text.splitlines()[0][:80]
        analysis.issues.append(
            Issue(
                category="Content: Untagged Text",
                line=1,
                message="Text detected outside of any <user> or <partner> tags",
                snippet=f"'{snippet}...'",
            )
        )

    # 5. Extract turns and validate contents & sequence
    turn_pattern = re.compile(
        r"<(user|partner|partner simple translation|partner simple formatting)>(.*?)</(user|partner|partner simple translation|partner simple formatting)>",
        re.DOTALL,
    )

    matches = list(turn_pattern.finditer(content))
    turns: list[dict[str, Any]] = []

    for m in matches:
        tag = m.group(1)
        text = m.group(2).strip()
        start_char = m.start()
        # Find line number from character index
        line_no = content[:start_char].count("\n") + 1
        turns.append({
            "tag": tag,
            "text": text,
            "line": line_no,
        })

        if tag == "user":
            analysis.turn_counts["user"] += 1
        elif tag == "partner":
            analysis.turn_counts["partner"] += 1
        elif tag in ("partner simple translation", "partner simple formatting"):
            analysis.turn_counts["partner_translation"] += 1

    analysis.total_turns = len(turns)

    # Check for empty turns
    for t in turns:
        if not t["text"]:
            analysis.issues.append(
                Issue(
                    category="Content: Empty Turn",
                    line=t["line"],
                    message=f"Tag '<{t['tag']}>' contains empty or whitespace-only text",
                )
            )

    # 6. Validate turn transitions
    # Expected sequence:
    #   [<user>]
    #   <partner> -> <partner simple translation|formatting>
    #   <partner simple translation|formatting> -> <user> (or EOF)
    #   <user> -> <partner> (or EOF)
    for idx, turn in enumerate(turns):
        tag = turn["tag"]
        line_no = turn["line"]
        next_turn = turns[idx + 1] if idx + 1 < len(turns) else None

        if tag == "user":
            if next_turn is not None:
                if next_turn["tag"] != "partner":
                    analysis.issues.append(
                        Issue(
                            category="Sequence: Unexpected Turn Transition",
                            line=line_no,
                            message=(
                                f"<user> at line {line_no} is followed by <{next_turn['tag']}> at line {next_turn['line']}. "
                                "Expected: <partner>"
                            ),
                            snippet=turn["text"].splitlines()[0][:80],
                        )
                    )
        elif tag == "partner":
            if next_turn is None:
                analysis.issues.append(
                    Issue(
                        category="Sequence: Incomplete Partner Turn",
                        line=line_no,
                        message=(
                            f"<partner> at line {line_no} is at end of file without translation/formatting. "
                            "Expected: <partner simple translation> or <partner simple formatting>"
                        ),
                        snippet=turn["text"].splitlines()[0][:80],
                    )
                )
            elif next_turn["tag"] not in ("partner simple translation", "partner simple formatting"):
                analysis.issues.append(
                    Issue(
                        category="Sequence: Missing Partner Translation",
                        line=line_no,
                        message=(
                            f"<partner> at line {line_no} is followed by <{next_turn['tag']}> at line {next_turn['line']}. "
                            "Expected: <partner simple translation> or <partner simple formatting>"
                        ),
                        snippet=turn["text"].splitlines()[0][:80],
                    )
                )
        elif tag in ("partner simple translation", "partner simple formatting"):
            if next_turn is not None:
                if next_turn["tag"] != "user":
                    analysis.issues.append(
                        Issue(
                            category="Sequence: Missing User Turn",
                            line=line_no,
                            message=(
                                f"<{tag}> at line {line_no} is directly followed by <{next_turn['tag']}> at line {next_turn['line']}. "
                                "Expected: <user> (or end of dialogue)"
                            ),
                            snippet=turn["text"].splitlines()[0][:80],
                        )
                    )

    # 7. Check for duplicated partner turns
    seen_partner_texts: dict[str, int] = {}
    for t in turns:
        if t["tag"] == "partner":
            norm_text = re.sub(r"\s+", " ", t["text"]).strip()
            if norm_text in seen_partner_texts:
                prev_line = seen_partner_texts[norm_text]
                analysis.issues.append(
                    Issue(
                        category="Content: Duplicate Partner Turn",
                        line=t["line"],
                        message=(
                            f"Duplicate <partner> turn detected at line {t['line']} (identical to line {prev_line})"
                        ),
                        snippet=t["text"].splitlines()[0][:80],
                    )
                )
            else:
                seen_partner_texts[norm_text] = t["line"]

    return analysis


def print_report(
    analyses: list[DialogAnalysis],
    verbose: bool = False,
) -> int:
    total_files = len(analyses)
    regular_files = [a for a in analyses if a.is_regular]
    irregular_files = [a for a in analyses if not a.is_regular]

    print("=" * 80)
    print(" DIALOG STRUCTURE ANALYSIS REPORT")
    print(" Expected Grammar: [<user>], <partner>, <partner translation>, <user>, ...")
    print("=" * 80)

    if irregular_files:
        print(f"\n[!] Found {len(irregular_files)} IRREGULAR dialog file(s) out of {total_files} total:\n")
        for a in irregular_files:
            print(f"--- FILE: {a.filepath.name} ({len(a.issues)} issue(s)) ---")
            print(f"    Path: {a.filepath}")
            print(
                f"    Turn Counts: {a.turn_counts['user']} user, "
                f"{a.turn_counts['partner']} partner, "
                f"{a.turn_counts['partner_translation']} translation/formatting"
            )
            print("    Issues:")
            for issue in a.issues:
                print(f"      - [Line {issue.line:3d}] [{issue.category}] {issue.message}")
                if issue.snippet:
                    print(f"        Snippet: \"{issue.snippet}\"")
            print()
    else:
        print(f"\n[OK] All {total_files} dialog files follow the regular turn structure!\n")

    if verbose and regular_files:
        print(f"--- REGULAR FILES ({len(regular_files)} files) ---")
        for a in regular_files:
            print(
                f"  [OK] {a.filepath.name:<30} ({a.total_turns:2d} tags: "
                f"{a.turn_counts['user']:2d} user, {a.turn_counts['partner']:2d} partner, "
                f"{a.turn_counts['partner_translation']:2d} trans)"
            )
        print()

    print("=" * 80)
    print(" SUMMARY")
    print("=" * 80)
    print(f" Total files analyzed  : {total_files}")
    print(f" Regular dialog files  : {len(regular_files)}")
    print(f" Irregular dialog files: {len(irregular_files)}")

    if irregular_files:
        print("\n List of irregular dialog files:")
        for idx, a in enumerate(irregular_files, 1):
            print(f"   {idx}. {a.filepath.name} ({len(a.issues)} issues)")
        print("\nResult: IRREGULAR DIALOGS DETECTED (Exit code 1)")
        return 1
    else:
        print("\nResult: ALL DIALOGS VALID (Exit code 0)")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze dialog turn regularity in raw dialog files."
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        default=Path("data/raw_dialogs"),
        help="Target directory containing raw dialog txt files (default: data/raw_dialogs)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed turn counts for regular files as well.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )
    args = parser.parse_args()

    if not args.dir.exists() or not args.dir.is_dir():
        print(f"[ERROR] Directory '{args.dir}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(2)

    files = sorted(args.dir.glob("*.txt"))
    if not files:
        print(f"[WARNING] No .txt files found in '{args.dir}'.", file=sys.stderr)
        sys.exit(0)

    analyses = [analyze_file(f) for f in files]

    if args.json:
        data = {
            "total_files": len(analyses),
            "regular_count": sum(1 for a in analyses if a.is_regular),
            "irregular_count": sum(1 for a in analyses if not a.is_regular),
            "analyses": [
                {
                    "file": a.filepath.name,
                    "path": str(a.filepath),
                    "is_regular": a.is_regular,
                    "turn_counts": a.turn_counts,
                    "total_turns": a.total_turns,
                    "issues": [issue.to_dict() for issue in a.issues],
                }
                for a in analyses
            ],
        }
        print(json.dumps(data, indent=2, ensure_ascii=False))
        sys.exit(1 if data["irregular_count"] > 0 else 0)
    else:
        exit_code = print_report(analyses, verbose=args.verbose)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
