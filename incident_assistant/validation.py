"""Input validation and best-effort redaction for incident text and logs."""

import os
import re
from dataclasses import dataclass

from .schemas import EvidenceRecord


MAX_UPLOAD_BYTES = 1024 * 1024
MAX_INCIDENT_CHARS = 10_000
MAX_LOG_EVIDENCE = 20
MAX_LOG_LINE_CHARS = 500

_SECRET_KEY_NAME = (
    r"(?:password|passwd|passphrase|pwd|secret|token|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|account[_-]?key|authorization|credentials?|signature)"
)
# Words that report a credential's status rather than its value ("password is incorrect",
# "authorization: denied", "token=expired"). They are kept so the diagnosis survives masking.
_STATUS_WORDS = (
    r"true|false|yes|no|null|none|expired|valid|invalid|incorrect|wrong|rejected|denied|"
    r"required|missing|empty|blank|reset|changed|locked|revoked|accepted|unknown|bad"
)
# Quoted values may contain escaped quotes ("pa\"ss"). Each alternative inside the quotes starts
# with a different character, so matching stays linear.
_DOUBLE_QUOTED = r'"(?:[^"\\\r\n]|\\.)*"'
_SINGLE_QUOTED = r"'(?:[^'\\\r\n]|\\.)*'"
# The key name may carry a bounded prefix/suffix (DB_PASSWORD, access_token, X-Auth-Token,
# spring.datasource.password). The lookbehind anchors matching to the start of a name token and
# the bounded repeats keep matching linear on long uploaded lines.
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)((?<![A-Za-z0-9_.-])[A-Za-z0-9_.-]{0,64}?"
    + _SECRET_KEY_NAME
    + r"[A-Za-z0-9_.-]{0,64}[\"']?\s*[:=]\s*)("
    + _DOUBLE_QUOTED
    + "|"
    + _SINGLE_QUOTED
    + r"|[^\s,;]+)"
)
# Command-line secrets: "--password hunter2", and MySQL's attached "-phunter2". The MySQL form is
# matched only after a MySQL client name, because "-p" usually means "port" elsewhere.
_CLI_SECRET_FLAG = re.compile(
    r"(?i)((?<![A-Za-z0-9_-])--?(?:password|passwd|pass|pwd|token|secret|api[_-]?key)\s+)(?!-)(\S+)"
)
_MYSQL_PASSWORD_FLAG = re.compile(
    r"(?i)(\b(?:mysql|mysqldump|mysqladmin|mariadb)\b[^\r\n]{0,200}?\s-p)([^\s-]\S*)"
)
# Short or generic key names (sig, sid, session IDs) are matched only as whole names, because as
# substrings they would also hit "signal=SIGKILL", "inside=..." and similar diagnostic fields.
_EXACT_SECRET_KEY = re.compile(
    r"(?i)((?<![A-Za-z0-9_.-])(?:sig|sid|sessionid|session[_-]id|jsessionid|phpsessid|"
    r"asp\.net_sessionid)[\"']?\s*[:=]\s*)("
    + _DOUBLE_QUOTED
    + "|"
    + _SINGLE_QUOTED
    + r"|[^\s,;&]+)"
)
_SECRET_IN_PROSE = re.compile(
    r"(?i)(\b(?:password|passphrase|passwd|pwd)\s+(?:is|was)\s+)"
    r"(?!(?:" + _STATUS_WORDS + r")\b)([^\s,;]+)"
)
_COOKIE_HEADER = re.compile(r"(?im)(\b(?:set-)?cookie\s*:\s*)[^\r\n]+")
_URL_CREDENTIALS = re.compile(
    # The username may be empty, as in redis://:password@host.
    r"(?i)((?<![a-z0-9+.-])[a-z][a-z0-9+.-]{0,31}://[^\s:/@]{0,256}:)[^\s/@]{1,256}@"
)
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----(?:.*?-----END [A-Z0-9 ]*PRIVATE KEY-----|.*\Z)",
    re.DOTALL,
)
# Mask the whole header value whatever the scheme (Bearer, Basic, Token, SSWS, ApiKey, ...):
# an optional scheme word plus the credential after it. A following "key=value" field is never
# taken as the credential, so "authorization=X path=/api status=500" keeps path and status.
# Digest and AWS SigV4 values hold comma-separated parameters, so they are masked to end of line.
_AUTHORIZATION = re.compile(
    r"(?i)((?<![A-Za-z0-9_.-])[A-Za-z0-9_.-]{0,32}?authorization[\"']?\s*[:=]\s*)("
    + _DOUBLE_QUOTED
    + "|"
    + _SINGLE_QUOTED
    + r"|(?:AWS4-HMAC-SHA256|Digest)\s[^\r\n]*"
    # A status word ("authorization: denied for user bob") is not a credential; leave it.
    + r"|(?!(?:" + _STATUS_WORDS + r")\b)"
    + r"(?:[A-Za-z][A-Za-z0-9._-]{0,30}[ \t]+(?![A-Za-z0-9_.-]+=[^=\s]))?[^\s,;]+)"
)
_BEARER = re.compile(r"(?i)(\bbearer\s+)([A-Za-z0-9._~+/=-]+)")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b")
# Anchored and bounded: an unanchored "\b[...]+@" rescans from every word boundary, which is
# quadratic on long dotted or hyphenated runs in a 1 MiB upload.
_EMAIL = re.compile(
    r"(?<![A-Z0-9._%+-])[A-Z0-9._%+-]{1,64}@[A-Z0-9.-]{1,253}\.[A-Z]{2,24}\b", re.IGNORECASE
)
_COMMON_TOKEN = re.compile(
    r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{22,}"
    r"|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|[sr]k_(?:live|test)_[A-Za-z0-9]{16,}"
    r"|AIza[0-9A-Za-z_-]{35}|glpat-[A-Za-z0-9_-]{20,}|npm_[A-Za-z0-9]{36})\b"
)
# Values that carry diagnostic meaning rather than a credential. They are kept only in these
# narrow cases; any other value under a secret-like key is masked.
_NON_SECRET_WORD = re.compile(r"(?i)^(?:" + _STATUS_WORDS + r")$")
_COUNT_KEY = re.compile(
    r"(?i)(?:max_?tokens|tokens?_(?:used|count|total|limit|remaining)|tokens"
    r"|_count|_days|_hours|_minutes|_seconds|_ms|_ttl|_length|_size|_limit)$"
)
_NUMBER_WITH_UNIT = re.compile(r"(?i)^\d+(?:\.\d+)?[a-z]{0,2}$")
_PATH_VALUE = re.compile(r"^(?:/|~|[A-Za-z]:[\\/])")
_KEY_NAME = re.compile(r"[A-Za-z0-9_.-]+")
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
    """Mask common credential patterns and email addresses; this is not exhaustive.

    Line count is preserved so log evidence line numbers still match the uploaded file.
    """
    text = _PRIVATE_KEY_BLOCK.sub(_replace_private_key, text)
    text = _AUTHORIZATION.sub(_replace_value, text)
    text = _BEARER.sub(r"\1[REDACTED]", text)
    text = _COOKIE_HEADER.sub(r"\1[REDACTED]", text)
    text = _URL_CREDENTIALS.sub(r"\1[REDACTED]@", text)
    text = _CLI_SECRET_FLAG.sub(r"\1[REDACTED]", text)
    text = _MYSQL_PASSWORD_FLAG.sub(r"\1[REDACTED]", text)
    text = _SECRET_ASSIGNMENT.sub(_replace_secret_assignment, text)
    text = _EXACT_SECRET_KEY.sub(_replace_secret_assignment, text)
    text = _SECRET_IN_PROSE.sub(r"\1[REDACTED]", text)
    text = _JWT.sub("[REDACTED_JWT]", text)
    text = _COMMON_TOKEN.sub("[REDACTED_TOKEN]", text)
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    return text


def _replace_private_key(match: re.Match[str]) -> str:
    # Keep the original newlines so later lines keep their line numbers.
    return "[REDACTED_PRIVATE_KEY]" + "\n" * match.group(0).count("\n")


def _is_diagnostic_value(key: str, value: str) -> bool:
    bare = value.strip("\"'")
    if _NON_SECRET_WORD.match(bare):
        return True
    if _COUNT_KEY.search(key) and _NUMBER_WITH_UNIT.match(bare):
        return True
    return key.lower() in {"pwd", "oldpwd"} and bool(_PATH_VALUE.match(bare))


def _replace_secret_assignment(match: re.Match[str]) -> str:
    key = _KEY_NAME.search(match.group(1))
    if key and _is_diagnostic_value(key.group(0), match.group(2)):
        return match.group(0)
    return _replace_value(match)


def _replace_value(match: re.Match[str]) -> str:
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


def is_error_line(text: str) -> bool:
    """True for log lines that report a failure (ERROR, FATAL, timeout, refused, 5xx, ...)."""
    return bool(_ERROR_LINE.search(text))


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
