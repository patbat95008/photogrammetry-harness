"""Source-file identity and path safety."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .config import get_settings
from .manifest import SourceIdentity

#: Hashing only the ends of the file. A full SHA-256 of a 4 GB clip takes ~15 s
#: and buys nothing over this: size plus the first and last chunk catches every
#: realistic case -- a replaced file, a re-export, a truncated transfer, a
#: different transcode. Two distinct videos agreeing on all three is not a
#: scenario worth 15 seconds on every page load.
CHUNK_BYTES = 8 * 1024 * 1024


def source_identity(path: Path) -> SourceIdentity:
    stat = path.stat()
    hasher = hashlib.sha256()
    hasher.update(str(stat.st_size).encode())

    with path.open("rb") as fh:
        hasher.update(fh.read(CHUNK_BYTES))
        if stat.st_size > CHUNK_BYTES * 2:
            fh.seek(-CHUNK_BYTES, 2)
            hasher.update(fh.read(CHUNK_BYTES))

    return SourceIdentity(
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        digest=hasher.hexdigest()[:32],
    )


def is_within_ingest_roots(path: Path) -> bool:
    """Whether a path is inside one of the configured ingest roots.

    The browse and clip-add endpoints are reachable from the page, so they must not
    be able to read arbitrary disk just because a URL said so.
    """
    resolved = path.resolve()
    for root in get_settings().ingest_roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except (ValueError, OSError):
            continue
    return False
