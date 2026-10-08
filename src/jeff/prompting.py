"""Prompt construction for SemIf-format decision rows.

Reimplements the prompt interface of SemIf (github.com/TheoLeeCJ/SemIf, MIT)
byte-for-byte: same system message, same JSON payload layout, same chat-template
arguments, same single-token answer-slot checks, same sha256 prompt digest.
Byte identity is verified in tests against the per-row prompt_sha256 values that
SemIf ships inside its committed prediction files.

A row is: {"id": str, "state": str|dict|list, "question": str,
           "options": [{"id": str, "description": str}, ...]}  (2..255 options)
Score-type (ordinal) rows use the same format: the ordered levels are the options.

Options are labeled A-P as in SemIf. Rows with more than 16 options continue
with Q-Z and then two-letter labels (LABELS, decider's scheme), so prompts of
rows with up to 16 options are unchanged.

Shorter prompt formats (opt-in, for the JevBench cost limit, which counts input
tokens per decision) are chosen with JEV_PROMPT_FORMAT, read once at import;
PROMPT_VERSION then names the active format, so feature caches built with
different prompts never mix (features.cache_meta records it):
  direct-options-v1  (default) SemIf: system message + JSON payload, byte-identical
  plain-v2           Quyet-1.0-Large's "prompt version 2" layout (no system message,
                     plain text, compact JSON states, one instruction line per type)
  compact-v3         ours: plain-v2 without the per-type instruction line and labels
"""

from __future__ import annotations

import hashlib
import json
import os
import string
import weakref

LETTERS = "ABCDEFGHIJKLMNOP"
MAX_OPTIONS = len(LETTERS)  # SemIf's width: default head slots and default data-builder width
# Answer labels for up to 255 options, as in decider (github.com/Mapika/decider,
# prompt.label_table): A-Z, then the first 229 two-letter uppercase strings that
# are one token in the Qwen3.5 tokenizer (the pairs below are not).
_NOT_ONE_TOKEN = set("BQ BZ CJ CQ CZ DQ DZ EJ EY FJ FQ FV FZ GJ GK GQ GZ HJ IY JF JG JH JL JN JQ".split())
LABELS = tuple(string.ascii_uppercase) + tuple(
    a + b for a in string.ascii_uppercase for b in string.ascii_uppercase if a + b not in _NOT_ONE_TOKEN)[:229]
MAX_LABELS = len(LABELS)  # 255: most options a row may have
PROMPT_FORMATS = ("direct-options-v1", "plain-v2", "compact-v3")
PROMPT_VERSION = "direct-options-v1"   # the format Jeff models were trained and evaluated with
if PROMPT_VERSION not in PROMPT_FORMATS:
    raise ValueError(f"JEV_PROMPT_FORMAT must be one of {PROMPT_FORMATS}, got {PROMPT_VERSION!r}")
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
# plain-v2: the per-type instruction lines of Quyet-1.0 (quyet/llm/prompt.py, Apache-2.0). Its yes/no line
# assumes option A is the "true" option, so it is used only when the first option says so.
PLAIN_KIND = {"choice": "Choose the option that fits best.",
              "score": "Choose the level that fits best (levels are ordered from lowest to highest).",
              "noul": "Choose A if the statement is true for this state, B if it is not."}


def validate_row(row: dict) -> None:
    required = {"id", "state", "question", "options"}
    if not required <= row.keys():
        raise ValueError(f"Row is missing fields: {sorted(required - row.keys())}")
    if not all(isinstance(row[key], str) and row[key] for key in ("id", "question")):
        raise ValueError("id and question must be nonempty strings")
    state = row["state"]
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValueError("state must be a nonempty string, object, or array")
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("state must be finite JSON-compatible data") from error
    options = row["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= MAX_LABELS:
        raise ValueError(f"options must contain 2-{MAX_LABELS} entries")
    ids = []
    for option in options:
        if not isinstance(option, dict) or not isinstance(option.get("id"), str) or not isinstance(option.get("description"), str):
            raise ValueError("Each option needs string id and description fields")
        ids.append(option["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Option IDs must be unique")


def direct_messages(row: dict) -> list[dict]:
    validate_row(row)
    payload = {
        "evidence": row["state"],
        "criterion": row["question"],
        "options": [
            {"letter": LABELS[index], "description": option["description"]}
            for index, option in enumerate(row["options"])
        ],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def state_text(state) -> str:
    """The state as plain text: strings unchanged, objects and lists as compact JSON."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def plain_messages(row: dict) -> list[dict]:
    """plain-v2: Quyet-1.0-Large's prompt version 2 layout, filled with our row's texts."""
    validate_row(row)
    qtype = row.get("qtype", "choice")
    if qtype == "noul" and row["options"][0]["id"].strip().lower() not in ("true", "yes"):
        qtype = "choice"
    options = "\n".join(f"{LABELS[i]}. {o['description']}" for i, o in enumerate(row["options"]))
    user = (f"State:\n{state_text(row['state'])}\n\nQuestion: {row['question']}\n{PLAIN_KIND[qtype]}\n\n"
            f"Options:\n{options}")
    return [{"role": "user", "content": user}]


def compact_messages(row: dict) -> list[dict]:
    """compact-v3: state, question and lettered options only; score rows say that the levels are ordered."""
    validate_row(row)
    ordered = " (levels from lowest to highest)" if row.get("qtype") == "score" else ""
    options = "\n".join(f"{LABELS[i]}. {o['description']}" for i, o in enumerate(row["options"]))
    return [{"role": "user", "content": f"{state_text(row['state'])}\n\nQuestion{ordered}: {row['question']}\n{options}"}]


def messages(row: dict) -> list[dict]:
    """Chat messages of a row in the active format (PROMPT_VERSION)."""
    if PROMPT_VERSION == "plain-v2":
        return plain_messages(row)
    if PROMPT_VERSION == "compact-v3":
        return compact_messages(row)
    return direct_messages(row)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def render_prompt(tokenizer, row: dict) -> str:
    return tokenizer.apply_chat_template(
        messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


_CACHE = weakref.WeakKeyDictionary()  # tokenizer -> checked slot ids, added tokens, checked tails


def _cache(tokenizer) -> dict:
    if tokenizer not in _CACHE:
        _CACHE[tokenizer] = {"slots": [], "added": list(tokenizer.get_added_vocab()), "tails": {}}
    return _CACHE[tokenizer]


def slot_ids(tokenizer, count: int) -> list[int]:
    """Token id of each answer label; each must be one exact round-trip token.
    Checked once per tokenizer and label, then cached."""
    if count > MAX_LABELS:
        raise ValueError(f"At most {MAX_LABELS} answer slots")
    known = _cache(tokenizer)["slots"]
    for label in LABELS[len(known):count]:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != label:
            raise ValueError(f"Answer slot {label!r} is not one exact round-trip token")
        known.append(encoded[0])
    result = known[:count]
    if len(result) != len(set(result)):
        raise ValueError("Answer-slot tokens collide")
    return result


def _tail(tokenizer, prompt: str) -> str | None:
    """Text after the last added token in the prompt (the chat template's
    "</think>\n\n"), or None if the prompt has no added token."""
    cut = max((prompt.rfind(token) + len(token) for token in _cache(tokenizer)["added"] if token in prompt),
              default=None)
    return None if cut is None else prompt[cut:]


def verify_boundary(tokenizer, prompt: str, ids: list[int], slots: list[int]) -> None:
    """SemIf's check: prompt + label re-tokenizes to ids + [label id], for every answer label.

    Rows with more than 16 options run the same check on the prompt's tail
    after its last added token, cached per tail. Tokenizers split added tokens
    out before BPE, so the result is the same, without up to 255 tokenizations
    of the whole prompt (seconds per wide row). Falls back to the full check if
    the prompt ids do not end with the tail's ids."""
    pairs = list(zip(LABELS, slots))
    tail = _tail(tokenizer, prompt) if len(slots) > len(LETTERS) else None
    if tail is not None:
        tail_ids = tokenizer.encode(tail, add_special_tokens=False)
        if ids[len(ids) - len(tail_ids):] == tail_ids:
            checked = _cache(tokenizer)["tails"]
            for label, token in pairs[checked.get(tail, 0):]:
                if tokenizer.encode(tail + label, add_special_tokens=False) != tail_ids + [token]:
                    raise ValueError(f"Answer boundary changes tokenization for slot {label}")
            if len(checked) < 1000:
                checked[tail] = max(checked.get(tail, 0), len(slots))
            return
    for label, token in pairs:
        if tokenizer.encode(prompt + label, add_special_tokens=False) != ids + [token]:
            raise ValueError(f"Answer boundary changes tokenization for slot {label}")


def encode_prompt(tokenizer, row: dict, max_tokens: int = 4096,
                  check_boundary: bool = True) -> tuple[list[int], list[int], str]:
    """Encode one decision row; returns (input_ids, slot_token_ids, prompt_sha256).

    Rows longer than max_tokens are rejected, not truncated (SemIf behavior).
    check_boundary re-tokenizes the prompt once per answer label to verify the
    label is read at a clean token boundary (SemIf's check; see verify_boundary
    for rows with more than 16 options); training loops run it on a sample only.
    """
    prompt = render_prompt(tokenizer, row)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > max_tokens:
        raise ValueError(f"Row {row['id']}: {len(ids)} input tokens exceed limit {max_tokens}; no truncation allowed")
    slots = slot_ids(tokenizer, len(row["options"]))
    if check_boundary:
        verify_boundary(tokenizer, prompt, ids, slots)
    return ids, slots, digest(prompt)

