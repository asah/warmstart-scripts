#!/usr/bin/env python3
"""
Step 1: Parse an mbox file and output a single markdown file.

Streams the mbox file incrementally — constant memory regardless of file size.
Handles 60GB+ mbox files without issue.

For each email, extracts:
- Interesting headers (From, To, Cc, Date, Subject, Message-ID, In-Reply-To,
  References, X-Gmail-Labels, List-Unsubscribe)
- Text body (prefers text/plain over text/html)

Output: one big markdown file with all emails separated by horizontal rules.
"""

import email
import email.policy
import sys
import os
import re
import html
from email.header import decode_header

# Headers worth keeping (lowercase for matching)
INTERESTING_HEADERS = [
    "from", "to", "cc", "bcc", "date", "subject",
    "message-id", "in-reply-to", "references",
    "x-gmail-labels", "reply-to", "list-unsubscribe",
]

# mbox "From " line pattern — starts a new message
# Format: "From <sender> <date>" at start of line
MBOX_FROM_RE = re.compile(rb"^From \S+.*\d{4}$|^From \S+.*\d{2}:\d{2}")


def decode_header_value(val):
    """Decode an RFC 2047 encoded header value to a string."""
    if val is None:
        return ""
    try:
        parts = decode_header(val)
        decoded = []
        for data, charset in parts:
            if isinstance(data, bytes):
                decoded.append(data.decode(charset or "utf-8", errors="replace"))
            else:
                decoded.append(data)
        return " ".join(decoded)
    except Exception:
        return str(val)


def get_text_body(msg):
    """Extract the text/plain body from a message. Falls back to text/html stripped of tags."""
    plain_parts = []
    html_parts = []

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if "attachment" in cd:
                continue
            if ct == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    plain_parts.append(payload.decode(charset, errors="replace"))
            elif ct == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    html_parts.append(payload.decode(charset, errors="replace"))
    else:
        ct = msg.get_content_type()
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            if ct == "text/plain":
                plain_parts.append(text)
            elif ct == "text/html":
                html_parts.append(text)

    if plain_parts:
        return "\n".join(plain_parts)
    if html_parts:
        return strip_html("\n".join(html_parts))
    return ""


def strip_html(text):
    """Crude HTML to text conversion."""
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?p[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?div[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_email(msg, idx):
    """Format a single email as a markdown section."""
    lines = []
    lines.append(f"## Email {idx}")
    lines.append("")

    for hdr in INTERESTING_HEADERS:
        values = msg.get_all(hdr, [])
        if values:
            name = hdr.replace("-", " ").title().replace(" ", "-")
            decoded = "; ".join(decode_header_value(v) for v in values)
            if len(decoded) > 500:
                decoded = decoded[:500] + "..."
            lines.append(f"**{name}:** {decoded}  ")

    lines.append("")

    body = get_text_body(msg)
    if body:
        lines.append("```")
        lines.append(body)
        lines.append("```")
    else:
        lines.append("*(no text body)*")

    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def stream_mbox(path):
    """
    Yield one raw message (bytes) at a time from an mbox file.

    Reads line-by-line — memory usage is O(single message), not O(file).
    Handles files of any size.

    Uses the strict mbox envelope format to avoid false splits on body lines
    that happen to start with "From ". The envelope line looks like:
      From sender@example.com Thu Apr 10 12:00:00 2026
    We require "From ", then a non-space token, then a timestamp-like pattern.
    """
    # Match: "From " + address/token + space + day-of-week (3 letters)
    # This is the POSIX mbox "From " line format. Body lines like
    # "From the beginning..." or "From: header" won't match.
    envelope_re = re.compile(
        rb"^From \S+ +"               # "From <addr> "
        rb"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)"  # day of week
    )

    buf = bytearray()

    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b"From ") and envelope_re.match(line):
                if buf:
                    yield bytes(buf)
                    buf.clear()
                # Skip the envelope line (not part of RFC 822 message)
                continue

            buf.extend(line)

    if buf:
        yield bytes(buf)


def main():
    mbox_path = sys.argv[1] if len(sys.argv) > 1 else "All mail Including Spam and Trash.mbox"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "emails.md"

    if not os.path.exists(mbox_path):
        print(f"Error: {mbox_path} not found", file=sys.stderr)
        sys.exit(1)

    file_size = os.path.getsize(mbox_path)
    file_size_gb = file_size / (1024 ** 3)
    print(f"Streaming {mbox_path} ({file_size_gb:.1f} GB)...", file=sys.stderr)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Email Archive\n\n")
        f.write(f"Source: `{os.path.basename(mbox_path)}`\n\n")
        f.write("---\n\n")

        count = 0
        errors = 0

        for raw_msg in stream_mbox(mbox_path):
            count += 1
            try:
                msg = email.message_from_bytes(raw_msg, policy=email.policy.compat32)
                f.write(format_email(msg, count))
            except Exception as e:
                errors += 1
                f.write(f"## Email {count}\n\n**Error parsing:** {e}\n\n---\n\n")

            if count % 1000 == 0:
                print(f"  {count} emails processed...", file=sys.stderr)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Done: {count} emails ({errors} errors) -> {out_path} ({size_mb:.1f} MB)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
