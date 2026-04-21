#!/usr/bin/env python3
"""
Step 2: Classify and summarize emails from emails.md

Architecture (tiered, cost-optimized):
  1. Regex — ONLY when absolutely certain (no false positives tolerated)
  2. SLM (qwen3-4b) — for the rest, with confidence score
  3. LLM (claude-haiku-4-5) — fallback when SLM confidence < threshold

Modes:
  (default)      Full tiered pipeline: regex → SLM → LLM fallback
  --offline      Regex-only pass (instant, free, conservative)
  --study N      Compare regex vs SLM vs LLM on N random emails
  --local        Use local ollama instead of Neurometric API for SLM

Local setup:
  brew install ollama
  ollama pull qwen3:4b
  python3 summarize_emails.py --local
"""

import re
import sys
import os
import json
import random
import asyncio
import argparse
from dataclasses import dataclass, field
from typing import Optional

try:
    import aiohttp
except ImportError:
    aiohttp = None

# ── Config ──────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("LITELLM_API_KEY", "")
API_BASE = os.environ.get("LITELLM_API_BASE", "https://api.neurometric.ai/v1")
LOCAL_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434/v1")

SLM_MODEL = os.environ.get("SLM_MODEL", "qwen3-4b")
SLM_MODEL_LOCAL = os.environ.get("SLM_MODEL_LOCAL", "qwen3:4b")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5")

SLM_CONFIDENCE_THRESHOLD = float(os.environ.get("SLM_CONFIDENCE", "0.85"))
MAX_CONCURRENT_REMOTE = int(os.environ.get("MAX_CONCURRENT", "10"))
MAX_CONCURRENT_LOCAL = int(os.environ.get("MAX_CONCURRENT_LOCAL", "1"))
MAX_BODY_CHARS = int(os.environ.get("MAX_BODY_CHARS", "3000"))
MAX_RETRIES = 3
RETRY_DELAY = 2

USE_LOCAL = False  # set via --local flag

# ── Classification taxonomy ─────────────────────────────────────────────────
CATEGORIES = [
    "relationship_reply_to_me",
    "relationship_1x1",
    "relationship_small_group",
    "list_mod_active",
    "mailing_list",
    "newsletter",
    "notification",
    "transactional",
    "calendar",
    "recruiting",
    "spam",
    "other",
]

CATEGORY_DESCRIPTIONS = {
    "relationship_reply_to_me": "Reply to something I (Adam) wrote — strongest relationship signal",
    "relationship_1x1": "Direct 1-on-1 email, personal or professional",
    "relationship_small_group": "Small group (2-5 people) discussion",
    "list_mod_active": "Active participant on a list I moderate",
    "mailing_list": "Mailing list / group email traffic",
    "newsletter": "Newsletter, marketing, or promotional content",
    "notification": "Automated notification or alert",
    "transactional": "Receipt, confirmation, shipping, or account activity",
    "calendar": "Calendar invite, RSVP, or meeting-related",
    "recruiting": "Job-related, recruiting, or hiring outreach",
    "spam": "Spam or unwanted bulk email",
    "other": "Doesn't fit other categories",
}

SLM_SYSTEM_PROMPT = f"""\
You are an email classifier and summarizer for the user's inbox. Output ONLY valid JSON (no markdown fences, no explanation) with these fields:

1. "category": exactly one of:
{chr(10).join(f'   - "{k}": {v}' for k, v in CATEGORY_DESCRIPTIONS.items())}

2. "confidence": float 0.0-1.0 — how certain you are. Use 0.95+ only when unambiguous. Use 0.5-0.7 for genuinely ambiguous cases.

3. "summary": 1-2 sentence summary of intent. Redact confidential details with [REDACTED].

4. "signature": contact details from signature block: {{name, title, company, phone, email, linkedin, website, address}}. Omit missing fields. null if no signature."""


@dataclass
class EmailRecord:
    idx: int
    headers: dict = field(default_factory=dict)
    body: str = ""


@dataclass
class ClassResult:
    idx: int
    category: str
    confidence: float
    summary: str
    signature: Optional[dict]
    source: str  # "regex", "slm", "llm"
    error: Optional[str] = None


def parse_markdown(filepath: str) -> list[EmailRecord]:
    """Parse emails.md into structured records (loads all into memory)."""
    records = list(stream_records(filepath))
    return records


def _parse_section(section: str) -> Optional[EmailRecord]:
    """Parse a single markdown section into an EmailRecord."""
    m = re.match(r"^## Email (\d+)", section)
    if not m:
        return None
    rec = EmailRecord(idx=int(m.group(1)))
    for hm in re.finditer(r"\*\*([^*]+):\*\*\s*(.+?)(?:\s\s|$)", section):
        rec.headers[hm.group(1).strip().lower()] = hm.group(2).strip()
    body_match = re.search(r"```\n(.*?)```", section, re.DOTALL)
    if body_match:
        rec.body = body_match.group(1).strip()
    return rec


def stream_records(filepath: str):
    """
    Yield EmailRecord objects one at a time by streaming the markdown file.
    Memory: O(single email section), not O(file).
    """
    buf = []
    in_email = False

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("## Email "):
                # Flush previous section
                if buf:
                    rec = _parse_section("".join(buf))
                    if rec is not None:
                        yield rec
                buf = [line]
                in_email = True
            elif in_email:
                buf.append(line)

    # Flush last section
    if buf:
        rec = _parse_section("".join(buf))
        if rec is not None:
            yield rec


# ── Tier 1: Conservative regex ──────────────────────────────────────────────
# ONLY classify when we are 100% certain. Return None otherwise.

# Domains/patterns that are DEFINITELY automated senders
DEFINITE_NOREPLY = re.compile(
    r"(^|<)(noreply|no-reply|donotreply|do-not-reply|mailer-daemon|postmaster)@",
    re.IGNORECASE,
)

# Definite calendar: iCalendar content or Google Calendar sender
DEFINITE_CALENDAR = re.compile(
    r"(text/calendar|\.ics\b|invite\.ics|BEGIN:VCALENDAR)", re.IGNORECASE
)
GCAL_SENDER = re.compile(r"calendar-notification@google\.com", re.IGNORECASE)

LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.IGNORECASE)
PHONE_RE = re.compile(r"[\+]?[\d\s\-\(\)]{7,15}")
URL_RE = re.compile(r"https?://[^\s<>\)\]\"]+")
EMAIL_RE = re.compile(r"[\w\.\-]+@[\w\.\-]+\.\w+")


def classify_regex(rec: EmailRecord) -> Optional[str]:
    """
    Conservative regex classifier. Returns a category ONLY when certain.
    Returns None when unsure — let SLM handle it.
    """
    from_addr = rec.headers.get("from", "").lower()
    to_addr = rec.headers.get("to", "").lower()
    subject = rec.headers.get("subject", "").lower()
    labels = rec.headers.get("x-gmail-labels", "").lower()
    body_lower = rec.body[:500].lower()  # only check start for speed

    # ── Calendar: iCal content or Google Calendar ──
    if GCAL_SENDER.search(from_addr):
        return "calendar"
    if DEFINITE_CALENDAR.search(rec.body[:1000]):
        return "calendar"

    # ── Transactional: noreply + receipt/order keywords ──
    if DEFINITE_NOREPLY.search(rec.headers.get("from", "")):
        transactional_kw = ["receipt", "order confirm", "shipping confirm",
                            "payment confirm", "invoice #", "your order"]
        if any(k in subject or k in body_lower for k in transactional_kw):
            return "transactional"
        # noreply but not transactional → notification (still safe)
        return "notification"

    # That's it. Everything else is ambiguous enough to need a model.
    return None


def extract_signature_regex(body: str) -> Optional[dict]:
    """Extract signature details heuristically."""
    sig_text = ""
    for delim in ["-- \n", "--\n", "Best,\n", "Thanks,\n", "Regards,\n",
                   "Cheers,\n", "Sincerely,\n", "Best regards,\n", "Thank you,\n",
                   "Warm regards,\n"]:
        idx = body.rfind(delim)
        if idx != -1 and idx > len(body) * 0.4:
            sig_text = body[idx:]
            break
    if not sig_text:
        sig_text = body[-500:] if len(body) > 500 else ""
    if not sig_text:
        return None

    sig = {}
    m = LINKEDIN_RE.search(sig_text)
    if m:
        url = m.group(0)
        sig["linkedin"] = url if url.startswith("http") else "https://" + url

    for p in PHONE_RE.findall(sig_text):
        if len(re.sub(r"[^\d\+]", "", p)) >= 10:
            sig["phone"] = p.strip()
            break

    emails = EMAIL_RE.findall(sig_text)
    if emails:
        sig["email"] = emails[0]

    for u in URL_RE.findall(sig_text):
        if "linkedin" not in u and "unsubscribe" not in u.lower():
            sig["website"] = u
            break

    return sig if sig else None


# ── Tier 2/3: Model API calls ──────────────────────────────────────────────

async def call_model(session, sem, rec: EmailRecord, model: str,
                     base_url: str, api_key: str) -> dict:
    """Call a model via OpenAI-compatible API. Returns parsed JSON or error."""
    subject = rec.headers.get("subject", "(no subject)")
    from_hdr = rec.headers.get("from", "unknown")
    to_hdr = rec.headers.get("to", "unknown")
    cc_hdr = rec.headers.get("cc", "")
    in_reply_to = rec.headers.get("in-reply-to", "")
    list_unsub = rec.headers.get("list-unsubscribe", "")
    labels = rec.headers.get("x-gmail-labels", "")
    body = rec.body[:MAX_BODY_CHARS] if rec.body else "(empty)"

    user_msg = f"""From: {from_hdr}
To: {to_hdr}
Cc: {cc_hdr or '(none)'}
Subject: {subject}
In-Reply-To: {in_reply_to or '(none)'}
List-Unsubscribe: {'yes' if list_unsub else 'no'}
X-Gmail-Labels: {labels}

Body:
{body}"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens": 400,
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
                        wait = int(resp.headers.get("Retry-After",
                                                     RETRY_DELAY * (attempt + 1)))
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    text = re.sub(r"^```(?:json)?\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)
                    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
                    # Find the JSON object in case of preamble
                    json_match = re.search(r"\{.*\}", text, re.DOTALL)
                    if json_match:
                        text = json_match.group(0)
                    parsed = json.loads(text)
                    if parsed.get("category") not in CATEGORIES:
                        parsed["category"] = "other"
                    return {"idx": rec.idx, "parsed": parsed, "error": None}
            except json.JSONDecodeError as e:
                return {"idx": rec.idx, "parsed": None, "error": f"JSON: {e}"}
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return {"idx": rec.idx, "parsed": None, "error": str(e)[:100]}

    return {"idx": rec.idx, "parsed": None, "error": "Max retries"}


# ── Tiered batch processing ─────────────────────────────────────────────────

def _make_regex_result(rec: EmailRecord, cat: str) -> ClassResult:
    """Build a ClassResult for a regex-classified email."""
    first_line = ""
    for line in rec.body.split("\n"):
        line = line.strip()
        if line and not line.startswith(">"):
            first_line = line[:200]
            break
    return ClassResult(
        idx=rec.idx, category=cat, confidence=1.0,
        summary=first_line or rec.headers.get("subject", ""),
        signature=extract_signature_regex(rec.body),
        source="regex",
    )


async def classify_batch(
    session: aiohttp.ClientSession,
    slm_sem: asyncio.Semaphore,
    llm_sem: asyncio.Semaphore,
    batch: list[EmailRecord],
) -> list[tuple[EmailRecord, ClassResult]]:
    """
    Classify a batch through all three tiers.
    Returns (record, result) pairs in input order.
    """
    results: dict[int, ClassResult] = {}
    need_slm: list[EmailRecord] = []

    # ── Tier 1: Regex ──
    for rec in batch:
        cat = classify_regex(rec)
        if cat is not None:
            results[rec.idx] = _make_regex_result(rec, cat)
        else:
            need_slm.append(rec)

    # ── Tier 2: SLM ──
    if need_slm:
        if USE_LOCAL:
            base_url, api_key = LOCAL_BASE, ""
            model = SLM_MODEL_LOCAL
            sem = slm_sem
        else:
            base_url, api_key = API_BASE, API_KEY
            model = SLM_MODEL
            sem = slm_sem

        need_llm: list[EmailRecord] = []
        tasks = [call_model(session, sem, rec, model, base_url, api_key)
                 for rec in need_slm]
        slm_results = await asyncio.gather(*tasks)

        for rec, r in zip(need_slm, slm_results):
            if r["error"] or r["parsed"] is None:
                need_llm.append(rec)
                continue
            p = r["parsed"]
            conf = float(p.get("confidence", 0))
            if conf >= SLM_CONFIDENCE_THRESHOLD:
                results[rec.idx] = ClassResult(
                    idx=rec.idx, category=p["category"], confidence=conf,
                    summary=p.get("summary", ""),
                    signature=p.get("signature"), source="slm",
                )
            else:
                need_llm.append(rec)

        # ── Tier 3: LLM fallback ──
        if need_llm:
            llm_tasks = [call_model(session, llm_sem, rec, LLM_MODEL,
                                    API_BASE, API_KEY)
                         for rec in need_llm]
            llm_results = await asyncio.gather(*llm_tasks)

            for rec, r in zip(need_llm, llm_results):
                if r["error"] or r["parsed"] is None:
                    results[rec.idx] = ClassResult(
                        idx=rec.idx, category="other", confidence=0.0,
                        summary=f"(classification error: {r.get('error', '')})",
                        signature=None, source="error",
                    )
                else:
                    p = r["parsed"]
                    results[rec.idx] = ClassResult(
                        idx=rec.idx, category=p["category"],
                        confidence=float(p.get("confidence", 0.9)),
                        summary=p.get("summary", ""),
                        signature=p.get("signature"), source="llm",
                    )

    return [(rec, results[rec.idx]) for rec in batch]


# ── Output formatting ───────────────────────────────────────────────────────

def format_summary(rec: EmailRecord, cr: ClassResult) -> str:
    lines = [f"## Email {rec.idx}", ""]

    # Injected classification header
    lines.append(f"**X-Classification:** {cr.category}  ")
    lines.append(f"**X-Classified-By:** {cr.source} (confidence: {cr.confidence:.2f})  ")

    for hdr in ["from", "to", "date", "subject", "x-gmail-labels"]:
        val = rec.headers.get(hdr)
        if val:
            name = hdr.replace("-", " ").title().replace(" ", "-")
            lines.append(f"**{name}:** {val}  ")

    lines.append("")
    lines.append(f"**Summary:** {cr.summary}")
    lines.append("")

    if cr.signature and isinstance(cr.signature, dict):
        parts = [f"{k}: {v}" for k, v in cr.signature.items() if v]
        if parts:
            lines.append(f"**Signature:** {' | '.join(parts)}")
            lines.append("")

    if cr.error:
        lines.append(f"**Error:** {cr.error}")
        lines.append("")

    lines.extend(["---", ""])
    return "\n".join(lines)


# ── Study mode ──────────────────────────────────────────────────────────────

async def run_study(input_path: str, n: int):
    records = parse_markdown(input_path)
    sample = random.sample(records, min(n, len(records)))
    print(f"Running classification study on {len(sample)} emails...\n", file=sys.stderr)

    sem = asyncio.Semaphore(MAX_CONCURRENT_REMOTE)
    slm_results = {}
    llm_results = {}

    async with aiohttp.ClientSession() as session:
        slm_tasks = [call_model(session, sem, rec, SLM_MODEL, API_BASE, API_KEY)
                     for rec in sample]
        llm_tasks = [call_model(session, sem, rec, LLM_MODEL, API_BASE, API_KEY)
                     for rec in sample]
        all_results = await asyncio.gather(*(slm_tasks + llm_tasks))
        for r in all_results[:len(sample)]:
            slm_results[r["idx"]] = r
        for r in all_results[len(sample):]:
            llm_results[r["idx"]] = r

    hdr = (f"{'Email':>6} | {'Regex':<28} | "
           f"{'SLM (' + SLM_MODEL + ')':<28} | "
           f"{'LLM (' + LLM_MODEL + ')':<28} | Match")
    print(hdr)
    print("-" * len(hdr))

    agree_all = agree_slm_llm = agree_rx_llm = 0
    slm_confs = []

    for rec in sample:
        rx = classify_regex(rec) or "UNCERTAIN"
        sr = slm_results.get(rec.idx, {})
        lr = llm_results.get(rec.idx, {})

        if sr.get("parsed"):
            sc = sr["parsed"].get("category", "ERROR")
            sconf = sr["parsed"].get("confidence", 0)
            slm_confs.append(sconf)
        else:
            sc, sconf = "ERROR", 0

        lc = lr["parsed"].get("category", "ERROR") if lr.get("parsed") else "ERROR"

        if rx == sc == lc:
            agree_all += 1
        if sc == lc:
            agree_slm_llm += 1
        if rx == lc:
            agree_rx_llm += 1

        match = "ALL" if rx == sc == lc else ("SLM=LLM" if sc == lc else "DIFFER")
        subj = rec.headers.get("subject", "")[:35]
        print(f"{rec.idx:>6} | {rx:<28} | {sc:<20} ({sconf:.2f}) | {lc:<28} | {match}")

    nt = len(sample)
    print(f"\n--- Agreement (n={nt}) ---")
    print(f"All three agree:  {agree_all}/{nt} ({100*agree_all/nt:.0f}%)")
    print(f"SLM = LLM:        {agree_slm_llm}/{nt} ({100*agree_slm_llm/nt:.0f}%)")
    print(f"Regex = LLM:      {agree_rx_llm}/{nt} ({100*agree_rx_llm/nt:.0f}%)")
    if slm_confs:
        print(f"SLM avg confidence: {sum(slm_confs)/len(slm_confs):.2f}")

    # Escalation analysis
    would_escalate = sum(1 for c in slm_confs if c < SLM_CONFIDENCE_THRESHOLD)
    print(f"\nWith threshold={SLM_CONFIDENCE_THRESHOLD}: {would_escalate}/{nt} "
          f"({100*would_escalate/nt:.0f}%) would escalate to LLM")


# ── Main pipeline (streaming) ───────────────────────────────────────────────

BATCH_SIZE_REMOTE = 50
BATCH_SIZE_LOCAL = 5

async def run(input_path: str, output_path: str, offline: bool):
    print(f"Streaming {input_path}...", file=sys.stderr)

    cat_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    total = 0

    mode = "offline-regex" if offline else f"tiered (regex→{SLM_MODEL}→{LLM_MODEL})"

    with open(output_path, "w", encoding="utf-8") as f:
        # Write header (stats will be appended at the end)
        f.write("# Email Archive — Summarized\n\n")
        f.write(f"Source: `{os.path.basename(input_path)}`  \n")
        f.write(f"Pipeline: `{mode}`  \n")
        f.write(f"Confidence threshold: {SLM_CONFIDENCE_THRESHOLD}\n\n")
        f.write("---\n\n")

        if offline:
            print("OFFLINE mode: conservative regex only", file=sys.stderr)
            for rec in stream_records(input_path):
                total += 1
                cat = classify_regex(rec) or "other"
                src = "regex" if cat != "other" else "unclassified"
                cr = _make_regex_result(rec, cat) if cat != "other" else ClassResult(
                    idx=rec.idx, category="other", confidence=0.0,
                    summary=_first_line(rec) or rec.headers.get("subject", ""),
                    signature=extract_signature_regex(rec.body),
                    source="unclassified",
                )
                f.write(format_summary(rec, cr))
                cat_counts[cr.category] = cat_counts.get(cr.category, 0) + 1
                source_counts[cr.source] = source_counts.get(cr.source, 0) + 1
                if total % 1000 == 0:
                    print(f"  {total} emails processed...", file=sys.stderr)
        else:
            if aiohttp is None:
                print("Error: pip install aiohttp", file=sys.stderr)
                sys.exit(1)

            batch_size = BATCH_SIZE_LOCAL if USE_LOCAL else BATCH_SIZE_REMOTE
            slm_sem = asyncio.Semaphore(
                MAX_CONCURRENT_LOCAL if USE_LOCAL else MAX_CONCURRENT_REMOTE)
            llm_sem = asyncio.Semaphore(MAX_CONCURRENT_REMOTE)

            async with aiohttp.ClientSession() as session:
                batch: list[EmailRecord] = []

                for rec in stream_records(input_path):
                    batch.append(rec)

                    if len(batch) >= batch_size:
                        pairs = await classify_batch(
                            session, slm_sem, llm_sem, batch)
                        for rec_out, cr in pairs:
                            f.write(format_summary(rec_out, cr))
                            cat_counts[cr.category] = cat_counts.get(cr.category, 0) + 1
                            source_counts[cr.source] = source_counts.get(cr.source, 0) + 1
                        total += len(batch)
                        batch = []
                        errors = source_counts.get("error", 0)
                        print(f"  {total} processed ({errors} errors)...",
                              file=sys.stderr)

                # Flush remaining
                if batch:
                    pairs = await classify_batch(
                        session, slm_sem, llm_sem, batch)
                    for rec_out, cr in pairs:
                        f.write(format_summary(rec_out, cr))
                        cat_counts[cr.category] = cat_counts.get(cr.category, 0) + 1
                        source_counts[cr.source] = source_counts.get(cr.source, 0) + 1
                    total += len(batch)

        # Append stats at the end (known only after streaming completes)
        f.write("\n---\n\n")
        f.write(f"## Summary Statistics\n\n")
        f.write(f"**Total emails:** {total}\n\n")

        f.write("### Classification Summary\n\n")
        f.write("| Category | Count | % |\n|---|---|---|\n")
        for cat in CATEGORIES:
            c = cat_counts.get(cat, 0)
            if c:
                f.write(f"| {cat} | {c} | {100*c/total:.1f}% |\n")
        f.write("\n")

        f.write("### Pipeline Stats\n\n")
        f.write("| Tier | Count | % |\n|---|---|---|\n")
        for src in ["regex", "slm", "llm", "error", "unclassified"]:
            c = source_counts.get(src, 0)
            if c:
                f.write(f"| {src} | {c} | {100*c/total:.1f}% |\n")
        f.write("\n")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    errors = source_counts.get("error", 0)
    print(f"Done: {total} emails ({errors} errors) -> {output_path} ({size_mb:.1f} MB)",
          file=sys.stderr)


def _first_line(rec: EmailRecord) -> str:
    """Extract the first non-empty, non-quoted line from the body."""
    for line in rec.body.split("\n"):
        line = line.strip()
        if line and not line.startswith(">"):
            return line[:200]
    return ""


def main():
    global USE_LOCAL, SLM_MODEL, LLM_MODEL, API_BASE, API_KEY
    global SLM_CONFIDENCE_THRESHOLD, MAX_CONCURRENT_REMOTE

    parser = argparse.ArgumentParser(
        description="Classify and summarize emails (tiered: regex → SLM → LLM)")
    parser.add_argument("input", nargs="?", default="emails.md")
    parser.add_argument("output", nargs="?", default="emails_summarized.md")
    parser.add_argument("--offline", action="store_true",
                        help="Conservative regex only (instant, free)")
    parser.add_argument("--local", action="store_true",
                        help="Use local ollama for SLM tier")
    parser.add_argument("--study", type=int, metavar="N",
                        help="Compare regex vs SLM vs LLM on N samples")
    parser.add_argument("--slm-model", help="Override SLM model")
    parser.add_argument("--llm-model", help="Override LLM model")
    parser.add_argument("--threshold", type=float,
                        help=f"SLM confidence threshold (default {SLM_CONFIDENCE_THRESHOLD})")
    parser.add_argument("--concurrency", type=int, help="Max concurrent requests")
    args = parser.parse_args()

    USE_LOCAL = args.local
    if args.slm_model:
        SLM_MODEL = args.slm_model
    if args.llm_model:
        LLM_MODEL = args.llm_model
    if args.threshold is not None:
        SLM_CONFIDENCE_THRESHOLD = args.threshold
    if args.concurrency:
        MAX_CONCURRENT_REMOTE = args.concurrency

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found. Run mbox_to_markdown.py first.",
              file=sys.stderr)
        sys.exit(1)

    if not args.offline and not args.local and not API_KEY:
        print("Error: set LITELLM_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    if args.study:
        if aiohttp is None:
            print("Error: pip install aiohttp", file=sys.stderr)
            sys.exit(1)
        asyncio.run(run_study(args.input, args.study))
    else:
        asyncio.run(run(args.input, args.output, args.offline))


if __name__ == "__main__":
    main()
