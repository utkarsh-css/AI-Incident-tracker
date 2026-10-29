"""Input validation and best-effort redaction for incident text and logs."""

import os
import re
from dataclasses import dataclass

from .schemas import EvidenceRecord


MAX_UPLOAD_BYTES = 1024 * 1024
MAX_INCIDENT_CHARS = 10_000
MAX_LOG_EVIDENCE = 20
MAX_LOG_LINE_CHARS = 500

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:password|passwd|pwd|api[_-]?key|access[_-]?key|secret|token|client[_-]?secret|authorization)\b[\"']?\s*[:=]\s*)(\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;]+)"
)
_AUTHORIZATION = re.compile(r"(?i)(\bauthorization\s*[:=]\s*)(?:basic|bearer)\s+[^\s,;]+")
_BEARER = re.compile(r"(?i)(\bbearer\s+)([A-Za-z0-9._~+/=-]+)")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_COMMON_TOKEN = re.compile(
    r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,})\b"
)
_TOKEN = re.compile(r"[a-z0-9_./-]+", re.IGNORECASE)
_ERROR_LINE = re.compile(
    r"(?i)\b(error|exception|failed|failure|timeout|timed out|refused|denied|critical|fatal| 5\d\d )\b"
)


@dataclass(frozen=True)
class ValidatedUpload:
    filename: str
    text: str


class InputValidationError(ValueError):
    """A safe, user-displayable input validation error."""


def validate_incident_description(description: str) -> str:
    cleaned = description.strip()
    if not cleaned:
        raise InputValidationError("Enter an incident description before analysis.")
    if len(cleaned) > MAX_INCIDENT_CHARS:
        raise InputValidationError("The incident description exceeds the 10,000 character limit.")
    return cleaned


def redact_sensitive_text(text: str) -> str:
    """Mask common credential patterns and email addresses; this is not exhaustive."""
    text = _AUTHORIZATION.sub(r"\1[REDACTED]", text)
    text = _BEARER.sub(r"\1[REDACTED]", text)
    text = _SECRET_ASSIGNMENT.sub(_replace_secret_assignment, text)
    text = _JWT.sub("[REDACTED_JWT]", text)
    text = _COMMON_TOKEN.sub("[REDACTED_TOKEN]", text)
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    return text


def _replace_secret_assignment(match: re.Match[str]) -> str:
    value = match.group(2)
    if value.startswith('"') and value.endswith('"'):
        replacement = '"[REDACTED]"'
    elif value.startswith("'") and value.endswith("'"):
        replacement = "'[REDACTED]'"
    else:
        replacement = "[REDACTED]"
    return match.group(1) + replacement


def validate_uploaded_log(filename: str, raw_bytes: bytes) -> ValidatedUpload:
    basename = os.path.basename(filename.replace("\\", "/")) or "uploaded log"
    basename = re.sub(r"[\x00-\x1f\x7f]", "", basename)
    extension = os.path.splitext(basename)[1]
    safe_stem = re.sub(r"[^A-Za-z0-9._ -]", "_", os.path.splitext(basename)[0])
    basename = f"{safe_stem[:120]}{extension[:8]}"
    if os.path.splitext(basename)[1].lower() not in {".log", ".txt"}:
        raise InputValidationError("Upload a .log or .txt file.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise InputValidationError("The uploaded file exceeds the 1 MiB size limit.")
    if not raw_bytes:
        raise InputValidationError("The uploaded file is empty.")
    try:
        decoded = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputValidationError("The uploaded file must be UTF-8 text.") from exc
    if not decoded.strip():
        raise InputValidationError("The uploaded file contains no readable text.")
    if "\x00" in decoded:
        raise InputValidationError("The uploaded file appears to contain binary data.")
    control_count = sum(ord(char) < 32 and char not in "\n\r\t\f" for char in decoded)
    if control_count / max(1, len(decoded)) > 0.02:
        raise InputValidationError("The uploaded file appears to contain binary data.")
    return ValidatedUpload(filename=basename, text=redact_sensitive_text(decoded))


def make_incident_evidence(description: str) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id="I1",
        kind="incident_description",
        source_name="Incident description",
        excerpt=description,
    )


def make_log_evidence(
    log: ValidatedUpload | None, incident_text: str
) -> list[EvidenceRecord]:
    """Create bounded, line-addressable references into sanitized uploaded text."""
    if log is None:
        return []

    query_terms = set(_TOKEN.findall(incident_text.lower()))
    candidates: list[tuple[int, int, str]] = []
    safe_text = redact_sensitive_text(log.text)
    for line_number, line in enumerate(safe_text.splitlines(), start=1):
        if not line.strip():
            continue
        terms = set(_TOKEN.findall(line.lower()))
        overlap = len(query_terms & terms)
        # Error-like lines get a small tie-breaking boost, not authority as a diagnosis.
        rank = overlap * 2 + int(bool(_ERROR_LINE.search(line)))
        candidates.append((rank, line_number, line))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    chosen = sorted(candidates[:MAX_LOG_EVIDENCE], key=lambda item: item[1])
    records = []
    for index, (_, line_number, line) in enumerate(chosen, start=1):
        excerpt = line[:MAX_LOG_LINE_CHARS]
        records.append(
            EvidenceRecord(
                evidence_id=f"L{index}",
                kind="uploaded_log",
                source_name=log.filename,
                excerpt=excerpt,
                line_start=line_number,
                line_end=line_number,
                truncated=len(line) > MAX_LOG_LINE_CHARS,
            )
        )
    return records
