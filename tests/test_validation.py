import pytest

from incident_assistant.validation import (
    InputValidationError,
    MAX_INCIDENT_CHARS,
    MAX_UPLOAD_BYTES,
    ValidatedUpload,
    make_log_evidence,
    redact_sensitive_text,
    validate_incident_description,
    validate_uploaded_log,
)


def test_incident_description_must_not_be_blank():
    with pytest.raises(InputValidationError, match="incident description"):
        validate_incident_description(" \n  ")


def test_incident_description_is_trimmed():
    assert validate_incident_description("  API is unavailable  ") == "API is unavailable"


def test_incident_description_has_a_documented_size_limit():
    with pytest.raises(InputValidationError, match="10,000 character"):
        validate_incident_description("x" * (MAX_INCIDENT_CHARS + 1))


@pytest.mark.parametrize("name", ["trace.pdf", "trace.csv", "trace"])
def test_rejects_unsupported_extensions(name):
    with pytest.raises(InputValidationError, match=r"\.log or \.txt"):
        validate_uploaded_log(name, b"some text")


def test_rejects_oversized_upload():
    with pytest.raises(InputValidationError, match="1 MiB"):
        validate_uploaded_log("trace.log", b"x" * (MAX_UPLOAD_BYTES + 1))


@pytest.mark.parametrize("content", [b"", b" \r\n\t"])
def test_rejects_empty_upload(content):
    with pytest.raises(InputValidationError):
        validate_uploaded_log("trace.txt", content)


@pytest.mark.parametrize("content", [b"\xff\xfe", b"a\x00b", b"\x01\x02\x03\x04\x05" * 10])
def test_rejects_binary_or_non_utf8_upload(content):
    with pytest.raises(InputValidationError):
        validate_uploaded_log("trace.log", content)


def test_accepts_utf8_and_uses_basename():
    result = validate_uploaded_log(r"C:\\private\\trace.LOG", b"database timeout\n")
    assert result.filename == "trace.LOG"
    assert result.text == "database timeout\n"


def test_redacts_common_secrets_bearer_jwt_and_email():
    source = (
        "password=hunter2 api_key:abc123 token=secret-value "
        "Authorization: Bearer abcdefghijklmnop "
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.signature123 "
        "alice@example.com"
    )
    result = redact_sensitive_text(source)
    assert "hunter2" not in result
    assert "abc123" not in result
    assert "secret-value" not in result
    assert "abcdefghijklmnop" not in result
    assert "alice@example.com" not in result
    assert "[REDACTED]" in result
    assert "[REDACTED_EMAIL]" in result


def test_redacts_quoted_json_values_authorization_and_common_key_shapes():
    source = (
        '{"password": "phrase with spaces", "api_key":"quoted-key"} '
        "Authorization: Basic dXNlcjpwYXNz "
        "sk-proj-1234567890abcdefghijklmnop"
    )
    result = redact_sensitive_text(source)
    assert "phrase with spaces" not in result
    assert "quoted-key" not in result
    assert "dXNlcjpwYXNz" not in result
    assert "sk-proj-1234567890abcdefghijklmnop" not in result
    assert '"password": "[REDACTED]"' in result


def test_log_evidence_is_sanitized_line_addressable_and_bounded():
    upload = ValidatedUpload(
        filename="system.log",
        text="noise\npassword=topsecret\nDatabase connection timeout\n" + ("warning " * 100),
    )
    records = make_log_evidence(upload, "Database connection timeout")
    assert records
    assert len(records) <= 20
    assert all(record.line_start == record.line_end for record in records)
    assert all(len(record.excerpt) <= 500 for record in records)
    assert records[-1].truncated
    assert "topsecret" not in " ".join(record.excerpt for record in records)
