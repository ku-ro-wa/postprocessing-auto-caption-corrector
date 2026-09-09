"""System prompt, few-shot examples, and reply parsing for the LLM pass.

The request is one chat turn: a fixed system message carrying four worked
examples (one per flag archetype -- phonetic-vocab hit, split word, OOV with no
candidate, not-an-error) and a user message with the batch as JSON. The reply
is expected to be a JSON array of ``{id, replacement, confidence, rationale}``.
"""

from __future__ import annotations

import json

from caption_checker.corrector import (
    Correction,
    CorrectorError,
    FlagContext,
    ensure_ids_match,
)

SYSTEM_PROMPT = """\
You correct errors that automatic speech recognition made in a lecture
transcript. You receive a JSON array of flagged spans. Each flag has the
suspicious span, the sentence it sits in, the detector's reason for flagging
it, ranked candidate corrections (possibly empty), the neighbouring cues, and
related spans elsewhere in the transcript that sound alike.

For each flag, decide the single best correction. Reply with ONLY a JSON array,
one object per flag, in the same order:

  {"id": "<the flag id>", "replacement": "<text>" | null,
   "confidence": <0..1>, "rationale": "<one short line>"}

Set "replacement" to null when the span is not actually an error. Keep
capitalisation and surrounding punctuation out of "replacement" -- return just
the corrected word or phrase. Do not add commentary outside the JSON array.

Examples:

Input: [{"id":"a","span":"cubernetes","sentence":"Next week we look at
cubernetes and container orchestration.","candidates":["Kubernetes"],
"detector":"phonetic_vocab","reason":"\\"cubernetes\\" sounds like domain term
\\"Kubernetes\\"","related":[]}]
Output: [{"id":"a","replacement":"Kubernetes","confidence":0.97,"rationale":"ASR
misspelling of the container platform"}]

Input: [{"id":"b","span":"con sensus","sentence":"Today we talk about con
sensus algorithms.","candidates":["consensus"],"detector":"split_word",
"reason":"\\"con sensus\\" joined sounds like \\"consensus\\"","related":[]}]
Output: [{"id":"b","replacement":"consensus","confidence":0.95,"rationale":"one
word split across two tokens"}]

Input: [{"id":"c","span":"rey","sentence":"The rey protocol tolerates
failures.","candidates":[],"detector":"oov","reason":"\\"rey\\" is not a common
word or known term","related":["Raft"]}]
Output: [{"id":"c","replacement":"Raft","confidence":0.6,"rationale":"garbled;
Raft fits the consensus context and appears elsewhere"}]

Input: [{"id":"d","span":"lease","sentence":"The node holds a lease on the
partition.","candidates":["least"],"detector":"phonetic_internal","reason":
"\\"lease\\" sounds like \\"least\\" used elsewhere","related":["least"]}]
Output: [{"id":"d","replacement":null,"confidence":0.9,"rationale":"\\"lease\\"
is correct here -- it is a distributed-systems term"}]
"""


def build_messages(batch: list[FlagContext]) -> list[dict]:
    user = json.dumps([fc.as_dict() for fc in batch], ensure_ascii=False)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_response(
    content: str, batch: list[FlagContext]
) -> list[Correction]:
    """Turn the model's text reply into one :class:`Correction` per flag.

    Raises :class:`CorrectorError` on anything the orchestrator should treat as
    a failed batch: non-JSON, wrong shape, or an id set that does not match the
    request exactly.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.index("[") :] if "[" in text else text

    try:
        raw = json.loads(text)
    except ValueError as exc:
        raise CorrectorError(f"reply is not JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise CorrectorError("reply is not a JSON array")

    seen: dict[str, Correction] = {}
    for item in raw:
        if not isinstance(item, dict) or "id" not in item:
            raise CorrectorError("reply item is not an object with an id")
        cid = str(item["id"])
        replacement = item.get("replacement")
        if replacement is not None:
            replacement = str(replacement)
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError) as exc:
            raise CorrectorError(f"bad confidence: {exc}") from exc
        seen[cid] = Correction(
            id=cid,
            replacement=replacement,
            confidence=max(0.0, min(1.0, confidence)),
            rationale=str(item.get("rationale", "")),
        )

    ensure_ids_match(set(seen), {fc.id for fc in batch})
    return [seen[fc.id] for fc in batch]
