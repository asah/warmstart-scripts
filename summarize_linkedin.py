#!/usr/bin/env python3
"""
Classify and summarize LinkedIn messages using a neurometric SLM.

Reads linkedin-messages.csv incrementally, calls qwen3-4b via Neurometric API
to classify and summarize the CONTENT field (with confidential info redacted),
and outputs a new CSV with the original key fields plus category and summary.

Usage:
  python3 summarize_linkedin.py [input.csv] [output.csv]
  python3 summarize_linkedin.py --concurrency 30
  python3 summarize_linkedin.py --local  # use local ollama
"""

import csv
import sys
import os
import re
import json
import asyncio
import argparse
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("Error: pip install aiohttp", file=sys.stderr)
    sys.exit(1)

# ── Config ──────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("LITELLM_API_KEY", "")
API_BASE = os.environ.get("LITELLM_API_BASE", "https://api.neurometric.ai/v1")
LOCAL_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434/v1")

SLM_MODEL = os.environ.get("SLM_MODEL", "qwen3-4b")
SLM_MODEL_LOCAL = os.environ.get("SLM_MODEL_LOCAL", "qwen3:4b")

MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "30"))
MAX_CONCURRENT_LOCAL = int(os.environ.get("MAX_CONCURRENT_LOCAL", "2"))
MAX_CONTENT_CHARS = int(os.environ.get("MAX_CONTENT_CHARS", "3000"))
MAX_RETRIES = 3
RETRY_DELAY = 2
BATCH_SIZE = 50

USE_LOCAL = False

# ── Classification taxonomy (LinkedIn-specific) ───────────────────────────
CATEGORIES = [
    "pitch_startup",
    "pitch_services",
    "recruiting",
    "networking",
    "collaboration",
    "event_invite",
    "introduction",
    "follow_up",
    "personal",
    "auto_responder",
    "spam",
    "other",
]

CATEGORY_DESCRIPTIONS = {
    "pitch_startup": "Startup founder pitching for investment, advice, or meeting",
    "pitch_services": "Selling professional services, SaaS, consulting, etc.",
    "recruiting": "Job opportunity, recruiting outreach, or hiring-related",
    "networking": "General networking, coffee chat, or 'let's connect' request",
    "collaboration": "Proposal for partnership, co-investment, or joint work",
    "event_invite": "Invitation to an event, conference, dinner, or meetup",
    "introduction": "Making or requesting an intro to someone else",
    "follow_up": "Following up on a previous conversation or meeting",
    "personal": "Personal message from a friend, colleague, or acquaintance",
    "auto_responder": "Automated reply or canned response",
    "spam": "Spam, mass outreach, or irrelevant bulk message",
    "other": "Doesn't fit other categories",
}

SLM_SYSTEM_PROMPT = f"""\
You are a LinkedIn message classifier and summarizer. Output ONLY valid JSON (no markdown fences, no explanation) with these fields:

1. "category": exactly one of:
{chr(10).join(f'   - "{k}": {v}' for k, v in CATEGORY_DESCRIPTIONS.items())}

2. "summary": 1-2 sentence tight summary of the message intent and key details. \
REDACT all confidential information: dollar amounts, revenue figures, valuations, \
specific financial terms, phone numbers, email addresses, and private URLs. \
Replace with [REDACTED]. Keep names and company names visible.

Output only the JSON object."""

# ── Fields to pass through ─────────────────────────────────────────────────
PASSTHROUGH_FIELDS = [
    "CONVERSATION ID",
    "CONVERSATION TITLE",
    "FROM",
    "SENDER PROFILE URL",
    "TO",
    "RECIPIENT PROFILE URLS",
    "DATE",
]

OUTPUT_FIELDS = PASSTHROUGH_FIELDS + ["CATEGORY", "SUMMARY"]


# ── SLM API call ──────────────────────────────────────────────────────────

async def call_slm(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                   row_idx: int, content: str, from_name: str, to_name: str,
                   model: str, base_url: str, api_key: str) -> dict:
    """Call SLM for a single message. Returns {idx, category, summary, error}."""
    user_msg = f"From: {from_name}\nTo: {to_name}\n\nMessage:\n{content[:MAX_CONTENT_CHARS]}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens": 300,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 429:
                        wait = int(resp.headers.get(
                            "Retry-After", RETRY_DELAY * (attempt + 1)))
                        print(f"  Rate limited, waiting {wait}s...", file=sys.stderr)
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    # Strip markdown fences and thinking tags
                    text = re.sub(r"^```(?:json)?\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)
                    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
                    json_match = re.search(r"\{.*\}", text, re.DOTALL)
                    if json_match:
                        text = json_match.group(0)
                    parsed = json.loads(text)
                    cat = parsed.get("category", "other")
                    if cat not in CATEGORIES:
                        cat = "other"
                    return {
                        "idx": row_idx,
                        "category": cat,
                        "summary": parsed.get("summary", ""),
                        "error": None,
                    }
            except json.JSONDecodeError as e:
                return {"idx": row_idx, "category": "other",
                        "summary": "(JSON parse error)", "error": str(e)}
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return {"idx": row_idx, "category": "other",
                            "summary": "(API error)", "error": str(e)[:100]}

    return {"idx": row_idx, "category": "other",
            "summary": "(max retries)", "error": "Max retries"}


# ── CSV streaming & batch processing ──────────────────────────────────────

def stream_rows(filepath: str):
    """Yield (row_index, row_dict) from CSV, one at a time."""
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            yield i, row


async def process_batch(session, sem, batch, model, base_url, api_key):
    """Process a batch of (idx, row) pairs concurrently. Returns list of augmented rows."""
    tasks = []
    for idx, row in batch:
        content = row.get("CONTENT", "") or ""
        from_name = row.get("FROM", "")
        to_name = row.get("TO", "")
        tasks.append(call_slm(session, sem, idx, content, from_name, to_name,
                              model, base_url, api_key))

    results = await asyncio.gather(*tasks)

    out_rows = []
    for (idx, row), result in zip(batch, results):
        out_row = {field: row.get(field, "") for field in PASSTHROUGH_FIELDS}
        out_row["CATEGORY"] = result["category"]
        out_row["SUMMARY"] = result["summary"]
        out_rows.append(out_row)
        if result["error"]:
            print(f"  Row {idx}: error: {result['error']}", file=sys.stderr)

    return out_rows


async def run(input_path: str, output_path: str):
    if USE_LOCAL:
        base_url, api_key = LOCAL_BASE, ""
        model = SLM_MODEL_LOCAL
        concurrency = MAX_CONCURRENT_LOCAL
    else:
        base_url, api_key = API_BASE, API_KEY
        model = SLM_MODEL
        concurrency = MAX_CONCURRENT

    sem = asyncio.Semaphore(concurrency)
    cat_counts: dict[str, int] = {}
    total = 0
    errors = 0

    print(f"Processing {input_path} -> {output_path}", file=sys.stderr)
    print(f"Model: {model} | Concurrency: {concurrency} | Batch: {BATCH_SIZE}",
          file=sys.stderr)

    with open(output_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        async with aiohttp.ClientSession() as session:
            batch = []

            for idx, row in stream_rows(input_path):
                batch.append((idx, row))

                if len(batch) >= BATCH_SIZE:
                    out_rows = await process_batch(
                        session, sem, batch, model, base_url, api_key)
                    for out_row in out_rows:
                        writer.writerow(out_row)
                        cat_counts[out_row["CATEGORY"]] = \
                            cat_counts.get(out_row["CATEGORY"], 0) + 1
                    total += len(batch)
                    batch = []
                    print(f"  {total} rows processed...", file=sys.stderr)

            # Flush remaining
            if batch:
                out_rows = await process_batch(
                    session, sem, batch, model, base_url, api_key)
                for out_row in out_rows:
                    writer.writerow(out_row)
                    cat_counts[out_row["CATEGORY"]] = \
                        cat_counts.get(out_row["CATEGORY"], 0) + 1
                total += len(batch)

    # Print stats
    print(f"\nDone: {total} messages -> {output_path}", file=sys.stderr)
    print(f"\nCategory breakdown:", file=sys.stderr)
    for cat in CATEGORIES:
        c = cat_counts.get(cat, 0)
        if c:
            print(f"  {cat:<20} {c:>5} ({100*c/total:.1f}%)", file=sys.stderr)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nOutput: {size_mb:.2f} MB", file=sys.stderr)


def main():
    global USE_LOCAL, SLM_MODEL, MAX_CONCURRENT, BATCH_SIZE

    parser = argparse.ArgumentParser(
        description="Classify and summarize LinkedIn messages via SLM")
    parser.add_argument("input", nargs="?", default="linkedin-messages.csv")
    parser.add_argument("output", nargs="?", default="linkedin_summarized.csv")
    parser.add_argument("--local", action="store_true",
                        help="Use local ollama instead of Neurometric API")
    parser.add_argument("--concurrency", type=int,
                        help=f"Max concurrent requests (default {MAX_CONCURRENT})")
    parser.add_argument("--batch", type=int,
                        help=f"Batch size (default {BATCH_SIZE})")
    parser.add_argument("--model", help="Override SLM model name")
    args = parser.parse_args()

    USE_LOCAL = args.local
    if args.model:
        SLM_MODEL = args.model
    if args.concurrency:
        MAX_CONCURRENT = args.concurrency
    if args.batch:
        BATCH_SIZE = args.batch

    if not args.local and not API_KEY:
        print("Error: set LITELLM_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(args.input, args.output))


if __name__ == "__main__":
    main()
