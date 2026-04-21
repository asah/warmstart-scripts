#!/usr/bin/env python3
"""
Filter emails.md or emails_summarized.md by header regex.

Streams input line-by-line — constant memory. Works with both formats
(raw markdown from mbox_to_markdown.py and summarized from summarize_emails.py).

Examples:
  # Emails TO adam
  python3 filter_emails.py emails.md --match 'To:.*adam' -o adam_emails.md

  # Exclude spam-labeled emails
  python3 filter_emails.py emails.md --exclude 'X-Gmail-Labels:.*Spam'

  # Only relationship emails (summarized format)
  python3 filter_emails.py emails_summarized.md --match 'X-Classification:.*relationship'

  # From a specific domain
  python3 filter_emails.py emails.md --match 'From:.*@sequoia\\.com'

  # Combine: from sequoia, not spam
  python3 filter_emails.py emails.md --match 'From:.*@sequoia' --exclude 'Labels:.*Spam'

  # Case-sensitive match
  python3 filter_emails.py emails.md --match 'Subject:.*URGENT' --case-sensitive
"""

import re
import sys
import os
import argparse


def stream_sections(filepath: str):
    """
    Yield (header_block, full_section) for each email.
    header_block = all **Header:** lines concatenated (for matching).
    full_section = the complete markdown section (for output).
    Streams line-by-line — constant memory.
    """
    buf = []
    headers = []

    def flush():
        if buf:
            return "\n".join(headers), "".join(buf)
        return None, None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("## Email "):
                h, s = flush()
                if s is not None:
                    yield h, s
                buf = [line]
                headers = []
            elif buf:
                buf.append(line)
                # Collect header lines (both **Bold:** format)
                if line.startswith("**") and ":**" in line:
                    # Strip markdown bold markers for cleaner matching
                    # "**From:** alice@example.com  " → "From: alice@example.com"
                    clean = line.replace("**", "").rstrip().rstrip(" ")
                    headers.append(clean)
            # Lines before the first "## Email" (file header) are skipped

    # Flush last section
    h, s = flush()
    if s is not None:
        yield h, s


def main():
    parser = argparse.ArgumentParser(
        description="Filter email markdown by header regex",
        epilog="Patterns match against header lines like 'From: ...', 'Subject: ...', etc.",
    )
    parser.add_argument("input", help="Input markdown file")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("-m", "--match", action="append", default=[],
                        help="Include only emails where a header matches this regex (repeatable, all must match)")
    parser.add_argument("-x", "--exclude", action="append", default=[],
                        help="Exclude emails where a header matches this regex (repeatable, any excludes)")
    parser.add_argument("--case-sensitive", action="store_true",
                        help="Make regex case-sensitive (default: case-insensitive)")
    parser.add_argument("-v", "--invert", action="store_true",
                        help="Invert the final match (output non-matching emails)")
    parser.add_argument("--count", action="store_true",
                        help="Just print the count of matching emails, don't output them")
    parser.add_argument("--headers-only", action="store_true",
                        help="Print only the headers of matching emails (no body/summary)")
    args = parser.parse_args()

    if not args.match and not args.exclude:
        parser.error("At least one --match or --exclude is required")

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    flags = 0 if args.case_sensitive else re.IGNORECASE

    # Pre-compile patterns
    match_pats = [re.compile(p, flags) for p in args.match]
    exclude_pats = [re.compile(p, flags) for p in args.exclude]

    matched = 0
    total = 0

    out = None
    if not args.count:
        if args.output:
            out = open(args.output, "w", encoding="utf-8")
        else:
            out = sys.stdout

        # Write a minimal header
        if args.output:
            out.write(f"# Filtered: {' '.join(args.match or [])} "
                      f"{'(excluding: ' + ' '.join(args.exclude) + ')' if args.exclude else ''}\n\n")
            out.write(f"Source: `{os.path.basename(args.input)}`\n\n---\n\n")

    for header_block, section in stream_sections(args.input):
        total += 1

        # All --match patterns must match at least one header line
        passes_match = all(
            pat.search(header_block) for pat in match_pats
        ) if match_pats else True

        # Any --exclude pattern matching means exclusion
        passes_exclude = not any(
            pat.search(header_block) for pat in exclude_pats
        ) if exclude_pats else True

        hit = passes_match and passes_exclude
        if args.invert:
            hit = not hit

        if hit:
            matched += 1
            if out is not None:
                if args.headers_only:
                    # Print the ## line and all **Header:** lines
                    for line in section.split("\n"):
                        if line.startswith("## Email ") or (line.startswith("**") and ":**" in line):
                            out.write(line + "\n")
                    out.write("\n---\n\n")
                else:
                    out.write(section)

    if out and out is not sys.stdout and args.output:
        out.close()

    print(f"{matched}/{total} emails matched", file=sys.stderr)


if __name__ == "__main__":
    main()
