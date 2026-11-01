"""multipart/form-data parsing (moved out of gateway.py, ERED LOMIN P1b).

Stdlib only. Python 3.13 has no cgi module, so the split is hand-rolled."""

import re


# --- Multipart parsing (Python 3.13 has no cgi module) --------------------

def parse_multipart_parts(body: bytes, content_type: str) -> dict[str, bytes]:
    """Hand-rolled multipart/form-data split. Returns {part_name: raw_bytes}
    for every named part found. A hand split is more predictable than
    email.parser for binary WAV payloads that may themselves contain
    "\\r\\n--" sequences (which is why we split on the exact boundary line,
    not on a generic pattern)."""
    m = re.search(r'boundary="?([^;"\s]+)"?', content_type or "")
    if not m:
        return {}
    boundary = m.group(1).encode()
    delim = b"--" + boundary
    parts = body.split(delim)
    out: dict[str, bytes] = {}
    for part in parts:
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        if b"\r\n\r\n" not in part:
            continue
        headers_blob, _, payload = part.partition(b"\r\n\r\n")
        # Negative lookbehind excludes "filename=" — Content-Disposition
        # headers often carry both "name=" and "filename=", and a plain
        # search for 'name="' would happily match inside "filename=" too.
        nm = re.search(rb'(?<!file)name="([^"]*)"', headers_blob)
        if not nm:
            continue
        # Trailing CRLF before the next boundary delimiter belongs to the
        # multipart framing, not the payload.
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        out[nm.group(1).decode("utf-8", "replace")] = payload
    return out


def parse_multipart_audio(body: bytes, content_type: str) -> bytes | None:
    """Returns the raw bytes of the part named "audio", or None if not
    found / not parseable."""
    return parse_multipart_parts(body, content_type).get("audio")
