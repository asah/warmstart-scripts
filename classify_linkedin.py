#!/usr/bin/env python3
"""
Parse and classify LinkedIn messages from the CSV export.

For each conversation, extracts:
- Participants (sender, receiver, profile URLs)
- Timestamps (first message, last message)
- Message count and direction breakdown
- Classification

Classifications:
  auto_responder_only  — only Adam's auto-responder, no real reply
  adam_engaged         — Adam sent substantive (non-auto) messages
  adam_initiated       — Adam sent the first message in the conversation
  inbound_unanswered  — someone messaged Adam, no reply at all
  inbound_spam        — in the SPAM folder
  group_chat          — 3+ participants
  other

Outputs markdown in the same format as the email pipeline, with
injected X-Classification headers, streamable.

Usage:
  python3 classify_linkedin.py [input.csv] [output.md]
  python3 classify_linkedin.py --stats  # just print classification stats
"""

import csv
import sys
import os
import re
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

OWNER_NAME = os.environ.get("OWNER_NAME", "Adam Sah")
OWNER_LINKEDIN_URL = os.environ.get("OWNER_LINKEDIN_URL", "https://www.linkedin.com/in/adamsah")

AUTO_RESPONDER_PREFIX = os.environ.get("AUTO_RESPONDER_PREFIX", "I'm flooded with LinkedIn msgs")

CATEGORIES = [
    "adam_initiated",
    "adam_engaged",
    "auto_responder_only",
    "inbound_unanswered",
    "inbound_spam",
    "group_chat",
    "other",
]


@dataclass
class Message:
    sender: str
    sender_url: str
    recipient: str
    recipient_urls: str
    date: str
    subject: str
    content: str
    folder: str
    is_auto_responder: bool


@dataclass
class Conversation:
    convo_id: str
    messages: list[Message] = field(default_factory=list)

    @property
    def participants(self) -> set[str]:
        names = set()
        for m in self.messages:
            names.add(m.sender)
            for r in m.recipient.split(","):
                r = r.strip()
                if r:
                    names.add(r)
        return names

    @property
    def other_participants(self) -> list[str]:
        return sorted(p for p in self.participants if p != OWNER_NAME)

    @property
    def other_profile_urls(self) -> list[str]:
        urls = set()
        for m in self.messages:
            if m.sender != OWNER_NAME and m.sender_url:
                urls.add(m.sender_url)
            for u in m.recipient_urls.split(","):
                u = u.strip()
                if u and u != OWNER_LINKEDIN_URL:
                    urls.add(u)
        return sorted(urls)

    @property
    def first_date(self) -> str:
        return min(m.date for m in self.messages) if self.messages else ""

    @property
    def last_date(self) -> str:
        return max(m.date for m in self.messages) if self.messages else ""

    @property
    def folder(self) -> str:
        folders = set(m.folder for m in self.messages)
        if "SPAM" in folders:
            return "SPAM"
        if "INBOX" in folders:
            return "INBOX"
        return next(iter(folders), "")

    @property
    def adam_messages(self) -> list[Message]:
        return [m for m in self.messages if m.sender == OWNER_NAME]

    @property
    def adam_substantive_messages(self) -> list[Message]:
        return [m for m in self.adam_messages if not m.is_auto_responder]

    @property
    def other_messages(self) -> list[Message]:
        return [m for m in self.messages if m.sender != OWNER_NAME]

    @property
    def first_sender(self) -> str:
        """Who sent the first message (chronologically)."""
        if not self.messages:
            return ""
        earliest = min(self.messages, key=lambda m: m.date)
        return earliest.sender


def parse_csv(filepath: str) -> dict[str, Conversation]:
    """Parse LinkedIn CSV into conversations, streaming."""
    convos: dict[str, Conversation] = {}

    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["CONVERSATION ID"]
            content = row.get("CONTENT", "") or ""
            is_auto = content.startswith(AUTO_RESPONDER_PREFIX)

            msg = Message(
                sender=row.get("FROM", ""),
                sender_url=row.get("SENDER PROFILE URL", ""),
                recipient=row.get("TO", ""),
                recipient_urls=row.get("RECIPIENT PROFILE URLS", ""),
                date=row.get("DATE", ""),
                subject=row.get("SUBJECT", ""),
                content=content,
                folder=row.get("FOLDER", ""),
                is_auto_responder=is_auto,
            )

            if cid not in convos:
                convos[cid] = Conversation(convo_id=cid)
            convos[cid].messages.append(msg)

    # Sort messages within each conversation chronologically
    for c in convos.values():
        c.messages.sort(key=lambda m: m.date)

    return convos


def classify(convo: Conversation) -> str:
    """Classify a conversation."""
    n_participants = len(convo.participants)

    if n_participants > 2:
        return "group_chat"

    if convo.folder == "SPAM":
        return "inbound_spam"

    has_adam_msgs = len(convo.adam_messages) > 0
    has_adam_substantive = len(convo.adam_substantive_messages) > 0
    has_other_msgs = len(convo.other_messages) > 0
    adam_started = convo.first_sender == OWNER_NAME

    if adam_started and has_adam_substantive:
        return "adam_initiated"

    if has_adam_substantive:
        return "adam_engaged"

    if has_adam_msgs and not has_adam_substantive:
        # Adam only sent auto-responders
        return "auto_responder_only"

    if has_other_msgs and not has_adam_msgs:
        return "inbound_unanswered"

    return "other"


def format_conversation(convo: Conversation, idx: int, category: str) -> str:
    """Format a conversation as a markdown section."""
    lines = [f"## Conversation {idx}", ""]

    lines.append(f"**X-Classification:** {category}  ")
    others = convo.other_participants
    lines.append(f"**From:** {', '.join(others) if others else '(unknown)'}  ")
    for url in convo.other_profile_urls:
        lines.append(f"**LinkedIn:** {url}  ")
    lines.append(f"**Date:** {convo.first_date} → {convo.last_date}  ")
    lines.append(f"**Messages:** {len(convo.messages)} "
                 f"({len(convo.adam_messages)} from Adam, "
                 f"{len(convo.adam_substantive_messages)} substantive)  ")
    lines.append(f"**Folder:** {convo.folder}  ")

    subject = next((m.subject for m in convo.messages if m.subject), "")
    if subject:
        lines.append(f"**Subject:** {subject}  ")

    lines.append("")

    # Show non-auto-responder messages (skip auto-responder noise)
    shown = 0
    for m in convo.messages:
        if m.is_auto_responder:
            continue
        content = m.content.replace("\xa0", " ").strip()
        if not content:
            continue
        # Truncate very long messages
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(f"**{m.sender}** ({m.date}):")
        lines.append(f"> {content}")
        lines.append("")
        shown += 1

    if shown == 0:
        if convo.adam_messages and all(m.is_auto_responder for m in convo.adam_messages):
            lines.append("*(auto-responder only, no substantive messages)*")
        else:
            lines.append("*(no message content)*")
        lines.append("")

    lines.extend(["---", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Classify LinkedIn messages")
    parser.add_argument("input", nargs="?", default="linkedin-messages.csv")
    parser.add_argument("output", nargs="?", default="linkedin_classified.md")
    parser.add_argument("--stats", action="store_true",
                        help="Print classification stats only")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {args.input}...", file=sys.stderr)
    convos = parse_csv(args.input)
    print(f"Found {len(convos):,} conversations, "
          f"{sum(len(c.messages) for c in convos.values()):,} messages",
          file=sys.stderr)

    # Classify all
    classified: list[tuple[Conversation, str]] = []
    cat_counts: dict[str, int] = defaultdict(int)

    for convo in sorted(convos.values(), key=lambda c: c.last_date, reverse=True):
        cat = classify(convo)
        classified.append((convo, cat))
        cat_counts[cat] += 1

    # Print stats
    print(f"\nClassification:", file=sys.stderr)
    for cat in CATEGORIES:
        c = cat_counts.get(cat, 0)
        print(f"  {cat:<25} {c:>5} ({100*c/len(classified):.1f}%)", file=sys.stderr)

    if args.stats:
        return

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# LinkedIn Messages — Classified\n\n")
        f.write(f"Source: `{os.path.basename(args.input)}`  \n")
        f.write(f"Conversations: {len(classified):,}  \n")
        f.write(f"Messages: {sum(len(c.messages) for c, _ in classified):,}  \n")
        f.write(f"Auto-responder instances: {sum(1 for c, _ in classified for m in c.messages if m.is_auto_responder):,}\n\n")

        f.write("### Classification Summary\n\n")
        f.write("| Category | Count | % |\n|---|---|---|\n")
        for cat in CATEGORIES:
            c = cat_counts.get(cat, 0)
            if c:
                f.write(f"| {cat} | {c:,} | {100*c/len(classified):.1f}% |\n")
        f.write("\n---\n\n")

        for idx, (convo, cat) in enumerate(classified, 1):
            f.write(format_conversation(convo, idx, cat))

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nDone: {len(classified):,} conversations -> {args.output} ({size_mb:.1f} MB)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
