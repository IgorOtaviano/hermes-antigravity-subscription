"""Prompt construction, message translation, and tool-call parsing for Antigravity."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import logging
import mimetypes
import os
import pathlib
import re
import socket
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from types import SimpleNamespace
from typing import Any, Callable, Optional

try:
    from hermes_constants import get_hermes_home
except ImportError:  # standalone use (tests, plugin dev)
    def get_hermes_home():
        return os.path.join(os.path.expanduser("~"), ".hermes")

logger = logging.getLogger(__name__)

# Media materialized for agy lives under the active Hermes home (profile-aware)
# and is deleted when the owning client session closes. The dir is resolved at
# USE time, not at import: one Hermes process can serve several profiles, and
# the cache must live under the home of the profile owning the session
# (review 5887913331, point 2).
def _media_cache_dir() -> str:
    return os.path.join(get_hermes_home(), "cache", "agy_media")


_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".ogv"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".oga", ".flac", ".opus", ".aac"}

# Remote fetch hardening: only public hosts, only media content types, and a
# hard size cap (the body is streamed, never read() whole).
_MAX_REMOTE_MEDIA_BYTES = 25 * 1024 * 1024
_MAX_DATA_URL_BYTES = 25 * 1024 * 1024

# mimetypes.guess_extension is OS-dependent (image/webp -> None on some
# systems); this explicit map keeps the extension gate deterministic.
_MEDIA_MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "video/x-msvideo": ".avi",
    "video/ogg": ".ogv",
}

# agy's headless media contract: pixels reach the model only through
# [Image](/abs/path) / [Video](/abs/path) markdown references in the turn text.
_MEDIA_REF_RE = re.compile(r"\[(?:Image|Video|Audio)\]\(([^)]+)\)")

_ATTACHMENT_ANNOTATION_RE = re.compile(
    # Bare [screenshot] (and any arg'd variant) carries no path: pure noise,
    # stripped as an annotation. [Image attached at: <path>] carries the ONLY
    # copy of the attachment path in a native-routed text message, so it is
    # NOT stripped here — _normalize_media_notes translates it into the
    # canonical reference. Stripping it first is what made native-routed
    # turns arrive at agy with no media reference at all.
    r"\n*\s*\[screenshot(?::\s*[^\]]+)?\]",
    re.IGNORECASE,
)


def _clean_rendered_text(text: str) -> str:
    if not text:
        return ""
    return _ATTACHMENT_ANNOTATION_RE.sub("", text).strip()


# --- Synthetic media notes -> canonical agy references -----------------------
#
# The Hermes core prepends several shapes of synthetic notes when a user turn
# carries media. In native routing the pixels travel as structured image_url
# parts; in text routing the turn text carries an auxiliary-vision description
# plus the local cache path. agy can only attach pixels from a markdown media
# reference, so every note shape is translated in place: the description block
# is dropped (the model receives the real pixels, a description of its own
# input is noise) and ONE canonical `[Image](/abs/path)` / `[Video](...)` is
# emitted where the note stood.

_NOTE_HEADS: tuple[tuple[str, str], ...] = (
    ("[The user sent a video attachment:", "Video"),
    ("[The user sent an audio file attachment:", "Audio"),
    ("[The user sent an image", "Image"),
    ("[The user attached an image", "Image"),
    ("[The tool result attached an image", "Image"),
    ("[The assistant attached an image", "Image"),
    ("[User sent an image:", "Image"),
    ("[Image attached at:", "Image"),
    # The paired "closer look" bracket that follows a description block (both
    # the gateway and the CLI variants). Consuming it here is what turns the
    # pair into exactly one reference instead of leaving the instruction
    # (and the path) as literal text the model can read.
    ("[If you need a closer look, use vision_analyze with", "Image"),
)

_VISION_NOTE_PATH_RE = re.compile(r"image_url:\s*([^\s\]~]+)")
_MEDIA_NOTE_SAVED_AT_RE = re.compile(r"It is saved at:\s*([^\s\]]+)")
_ATTACHED_AT_RE = re.compile(r"^\[Image attached at:\s*([^\]]+)\]\s*$")


def _consume_bracket_block(text: str, start: int) -> int:
    """Index just past the balanced ``[...]`` block opening at *start*, or -1.

    Balanced scanning instead of a lazy ``[^]]*`` regex: auxiliary vision
    descriptions may themselves contain ``]`` (markdown in the analysis), and
    the note only truly ends at depth zero.
    """
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


_MEDIA_LOCAL_EXTS = _IMAGE_EXTS | _VIDEO_EXTS | _AUDIO_EXTS


def _path_to_kind(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return "Video" if ext in _VIDEO_EXTS else "Image"


def _local_media_kind(path: str) -> Optional[str]:
    """Image/Video/Audio for a local path with an allowed media extension.

    Fail-closed gate: a local file without a known media extension is never
    handed to agy (agy uploads what it reads to Google).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTS:
        return "Image"
    if ext in _VIDEO_EXTS:
        return "Video"
    if ext in _AUDIO_EXTS:
        return "Audio"
    return None


def _media_reference_for(
    path: str, fallback_kind: str, media_sink: Optional[set] = None
) -> Optional[str]:
    """Canonical agy reference for a note-embedded path, or None.

    Per the normalization contract the reference is emitted by PATH, not by
    existence: a file that vanished (scratch pruning, replay from another
    host) must still surface as `[Image](/abs/path)` so the turn keeps its
    media intent instead of silently degrading to text-only. The extension
    gate applies regardless of existence: an unknown-extension path is
    dropped entirely.
    """
    if not path:
        return None
    kind = _local_media_kind(path)
    if kind is None:
        return None
    fpath, mtype = _materialize_media(path, media_sink)
    kind = mtype or kind
    return f"[{kind}]({fpath or path})"


def _normalize_media_notes(text: str, media_sink: Optional[set] = None) -> str:
    """Replace every synthetic media note with its canonical reference."""
    if not text or "[" not in text:
        return text
    out: list[str] = []
    emitted: set[str] = set()
    cursor = 0
    while True:
        hit: Optional[tuple[int, str, str]] = None
        for head, kind in _NOTE_HEADS:
            pos = text.find(head, cursor)
            if pos != -1 and (hit is None or pos < hit[0]):
                hit = (pos, head, kind)
        if hit is None:
            out.append(text[cursor:])
            break
        pos, head, kind = hit
        out.append(text[cursor:pos])
        end = _consume_bracket_block(text, pos)
        if end == -1:
            # Unterminated note head: keep the rest verbatim (an echo of the
            # marker in prose must not truncate the message).
            out.append(text[pos:])
            break
        block = text[pos:end]
        path: Optional[str] = None
        if head == "[Image attached at:":
            match = _ATTACHED_AT_RE.match(block.strip())
            path = match.group(1).strip() if match else None
        elif "image_url:" in block:
            match = _VISION_NOTE_PATH_RE.search(block)
            path = match.group(1).rstrip("~").strip() if match else None
        elif "saved at:" in block:
            match = _MEDIA_NOTE_SAVED_AT_RE.search(block)
            path = match.group(1).rstrip(".").strip() if match else None
        if path:
            reference = _media_reference_for(path, kind, media_sink)
            if reference and reference not in emitted:
                emitted.add(reference)
                out.append(reference)
        # Notes with no resolvable path are pure noise (failed analysis,
        # duplicated attachment marker): dropped, never re-described.
        cursor = end
    if not out:
        return text
    return "\n".join(part.strip("\n") for part in out if part and part.strip("\n")).strip() or ""


def _host_is_publicly_fetchable(hostname: str) -> bool:
    """True only when *hostname* resolves exclusively to public unicast addresses.

    SSRF gate for remote media fetches: loopback, private, link-local
    (which covers cloud metadata endpoints like 169.254.169.254), reserved,
    multicast and unspecified addresses make the host refuseable — every
    address the resolver returns counts, so a mixed-resolution hostname
    (one public, one private A record) is refused too.
    """
    if not hostname:
        return False
    try:
        infos = socket.getaddrinfo(hostname.rstrip("."), None)
    except (socket.gaierror, OSError, UnicodeError):
        return False
    if not infos:
        return False
    for info in infos:
        try:
            ip_text = str(info[4][0])
            address = ipaddress.ip_address(ip_text.split("%", 1)[0])
        except (ValueError, IndexError):
            return False
        if (
            address.is_loopback
            or address.is_private
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            return False
    return True


def _assert_fetchable_http_url(url: str) -> None:
    """Raise ValueError unless *url* is http(s) to a publicly resolvable host."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"refused media URL scheme: {parsed.scheme!r}")
    hostname = parsed.hostname or ""
    if not _host_is_publicly_fetchable(hostname):
        raise ValueError(f"refused non-public media host: {hostname!r}")


class _SafeMediaRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect hop exactly like the initial URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            _assert_fetchable_http_url(newurl)
        except ValueError as exc:
            raise urllib.error.URLError(f"media redirect refused: {exc}") from exc
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_MEDIA_OPENER = urllib.request.build_opener(_SafeMediaRedirectHandler)

# Materialized-by-URL memo: prompt assembly renders the same messages more
# than once per turn (permission collection, prompt assembly, latest-user
# extraction), and without a memo every render would re-download. Entries are
# hints: a path whose file was deleted (session close) re-downloads on next
# use instead of handing agy a dead reference.
_REMOTE_MEDIA_MEMO: dict[str, tuple[str, str]] = {}
_REMOTE_MEDIA_MEMO_LIMIT = 512


def _is_plugin_owned_media_path(path: str) -> bool:
    """True only for files this plugin wrote into its own media cache dir.

    Pass-through references (the user's own photos, Hermes' gateway media
    cache files, anything handed over by bare path or file:// URI) resolve
    OUTSIDE the dir: they go to agy by reference and must never be registered
    for deletion, or session close would delete the user's own data (review
    5887913331, point 1). realpath comparison: the cache dir or the file may
    sit behind a symlink, and realpath works on vanished paths too.
    """
    if not path:
        return False
    cache_root = os.path.realpath(_media_cache_dir())
    real = os.path.realpath(path)
    return real == cache_root or real.startswith(cache_root + os.sep)


def _register_materialized(
    fpath: str, mtype: str, media_sink: Optional[set]
) -> tuple[str, str]:
    # Only files the plugin itself wrote into its media cache dir enter the
    # close-deletion sink; pass-through user files never do.
    if fpath and media_sink is not None and _is_plugin_owned_media_path(fpath):
        media_sink.add(fpath)
    return fpath, mtype


def _media_ext_for_mime(mime: str) -> Optional[str]:
    ext = mimetypes.guess_extension(mime)
    if ext == ".jpe":
        ext = ".jpg"
    return ext or _MEDIA_MIME_EXT.get(mime)


def _mime_kind(mime: str) -> str:
    if mime.startswith("video/"):
        return "Video"
    if mime.startswith("audio/"):
        return "Audio"
    return "Image"


def unquote_path(path: str) -> str:
    return urllib.parse.unquote(path or "")


def _stream_download_to_cache(resp: Any, ext: str) -> Optional[tuple[str, str]]:
    """Stream a media response into the cache dir under a hard size cap.

    Writes to a temp file and renames only after the whole body passed the
    cap, so an aborted download never leaves a partial file behind for agy
    to read. Returns (path, kind) or None when the cap is exceeded.
    """
    digest = hashlib.sha256()
    cache_dir = _media_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".part")
    total = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_REMOTE_MEDIA_BYTES:
                    logger.debug("Remote media exceeds %d byte cap", _MAX_REMOTE_MEDIA_BYTES)
                    return None
                digest.update(chunk)
                out.write(chunk)
        fpath = os.path.join(cache_dir, f"{digest.hexdigest()[:16]}{ext}")
        os.replace(tmp_path, fpath)
        return fpath, _mime_kind(resp.headers.get_content_type().lower())
    except Exception:
        logger.debug("Streaming media download failed", exc_info=True)
        return None
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _materialize_media(
    url_or_data: str, media_sink: Optional[set] = None
) -> tuple[str, str]:
    """Materialize a media reference (data URL, local path, or remote URL)
    into a local file so the official `agy` CLI can resolve it natively.

    Every input is gated fail-closed:
    - data URLs: only image/* and video/* mimes, capped decoded size;
    - local paths and file:// URIs: only when the extension is a known
      media extension (nothing unknown is ever handed to agy, which
      uploads what it reads);
    - http(s) URLs: only media content types, streamed with a hard size
      cap, only to publicly resolvable hosts, re-validated on redirects.

    Returns:
        (local_file_path, "Image" | "Video" | "Audio") or ("", "") when the
        reference is refused or cannot be materialized.
    """
    url = (url_or_data or "").strip()
    if not url:
        return "", ""

    # 1. Base64 data URL
    if url.startswith("data:"):
        try:
            header, b64_data = url.split(",", 1)
            mime = header.split(";")[0].replace("data:", "").strip().lower()
            if not (mime.startswith("image/") or mime.startswith("video/") or mime.startswith("audio/")):
                logger.debug("Refused non-media data URL mime: %s", mime)
                return "", ""
            raw = base64.b64decode(b64_data)
            if len(raw) > _MAX_DATA_URL_BYTES:
                logger.debug("Refused oversized data URL media (%d bytes)", len(raw))
                return "", ""
            ext = _media_ext_for_mime(mime)
            if not ext:
                logger.debug("Refused data URL mime with no media extension: %s", mime)
                return "", ""
            h = hashlib.sha256(raw).hexdigest()[:16]
            cache_dir = _media_cache_dir()
            fpath = os.path.join(cache_dir, f"{h}{ext}")
            if not os.path.exists(fpath):
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    with open(fpath, "wb") as f:
                        f.write(raw)
                except OSError as exc:
                    logger.debug("Failed to write materialized data URL media: %s", exc)
                    return "", ""
            return _register_materialized(fpath, _mime_kind(mime), media_sink)
        except Exception as exc:
            logger.debug("Failed to materialize base64 media data URL: %s", exc)
            return "", ""

    # 2. file:// URI (local host only, no credentials/host indirection)
    if url.startswith("file://"):
        parsed = urllib.parse.urlsplit(url)
        if parsed.netloc not in ("", "localhost"):
            logger.debug("Refused file:// URI with host: %s", parsed.netloc)
            return "", ""
        local = unquote_path(parsed.path)
        kind = _local_media_kind(local)
        if kind is None:
            return "", ""
        # Emitted by path even when the file vanished: the media intent must
        # survive (agy's read of a missing file fails harmlessly).
        target_path = (
            local
            if local.startswith(("/", "\\")) or os.path.isabs(local)
            else os.path.abspath(local)
        )
        return _register_materialized(target_path, kind, media_sink)

    # 3. HTTP / HTTPS URL (SSRF-gated, capped, media content types only)
    if url.startswith(("http://", "https://")):
        try:
            _assert_fetchable_http_url(url)
        except ValueError as exc:
            logger.debug("Refused remote media URL %s: %s", url, exc)
            return "", ""
        memo = _REMOTE_MEDIA_MEMO.get(url)
        if memo is not None:
            if os.path.exists(memo[0]):
                return _register_materialized(memo[0], memo[1], media_sink)
            _REMOTE_MEDIA_MEMO.pop(url, None)
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Hermes-Agent"}
            )
            with _MEDIA_OPENER.open(req, timeout=15) as resp:
                content_type = resp.headers.get_content_type().lower()
                if not (
                    content_type.startswith("image/")
                    or content_type.startswith("video/")
                    or content_type.startswith("audio/")
                ):
                    logger.debug(
                        "Refused remote media content type %s for %s",
                        content_type,
                        url,
                    )
                    return "", ""
                ext = _media_ext_for_mime(content_type)
                if not ext:
                    logger.debug(
                        "Refused remote media content type with no extension: %s",
                        content_type,
                    )
                    return "", ""
                result = _stream_download_to_cache(resp, ext)
            if result is None:
                return "", ""
            if len(_REMOTE_MEDIA_MEMO) >= _REMOTE_MEDIA_MEMO_LIMIT:
                for stale in list(_REMOTE_MEDIA_MEMO)[: _REMOTE_MEDIA_MEMO_LIMIT // 2]:
                    _REMOTE_MEDIA_MEMO.pop(stale, None)
            _REMOTE_MEDIA_MEMO[url] = result
            return _register_materialized(result[0], result[1], media_sink)
        except Exception as exc:
            logger.debug("Failed to download remote media URL %s: %s", url, exc)
            return "", ""

    # 4. Local filesystem path (checked last: URL shapes never reach it).
    # Extension gate only; emitted by path even when the file vanished.
    kind = _local_media_kind(url)
    if kind is not None:
        target_path = (
            url
            if (
                url.startswith(("/", "\\"))
                or os.path.isabs(url)
                or pathlib.PureWindowsPath(url).is_absolute()
            )
            else os.path.abspath(url)
        )
        return _register_materialized(target_path, kind, media_sink)

    return "", ""

_ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool Result",
}

_PROMPT_PREAMBLE = (
    # agy embeds its own system prompt before the user's message. Some models
    # (claude-sonnet-4-6, gemini-3.6-flash) obey it and invoke native tools,
    # which headless mode soft-denies. A confrontational denial ("you are NOT
    # Antigravity") makes the model classify the preamble as prompt injection
    # and resist harder. This framing avoids contradicting agy's identity and
    # instead redirects the tool-calling protocol. (Issue #16)
    "You are operating in HEADLESS INFERENCE MODE. In this mode, your native tools (run_command, write_to_file, view_file, edit_file, etc.) are disabled and will be silently rejected. Do not attempt to use them.",
    "You are being used strictly as an LLM inference backend for Hermes Agent.",
    "All tool executions and filesystem interactions are performed exclusively by Hermes Agent on the host side.",
    "IMPORTANT INSTRUCTIONS FOR TOOLS:",
    "- If you need to call a tool, emit ONLY <tool_call>{...}</tool_call> blocks in your text output.",
    "- Each tool call must be a JSON object containing 'id', 'type': 'function', and 'function': {'name': '...', 'arguments': '...'}.",
    "- 'arguments' must be a JSON-encoded string containing the function arguments.",
    "- Do NOT execute local shell commands or file operations directly; only output <tool_call> tags so Hermes can execute them safely.",
    "- NEVER simulate or hallucinate Tool Results (e.g. 'Tool Result (...):'). Hermes Agent executes tools externally and will supply real results in subsequent turns.",
    "- After emitting tool calls, STOP generating immediately. Do NOT generate results, execution output, or commentary after tool calls.",
    "- If no tool is needed, respond naturally with standard text.",
    "- When the user's latest message introduces a new question or task, do NOT continue prior completed tasks or tool loops. Address the user's latest request directly.",
)

def _env_int(name: str, default: int) -> int:
    """An integer setting from the environment; a value that is not one keeps the default.

    Read at import, so a typo in the variable must not stop the plugin from loading.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring %s=%r: not an integer, using %d.", name, raw, default)
        return default


# --- Size limits of one message to agy ---------------------------------------
#
# agy cuts a single user message at a fixed size and drops everything after it,
# silently: the turn completes, exit code 0. Measured on agy 1.5.9 with
# gemini-3.8-flash-low and gemini-4-hydrogen-medium (same result on both): the
# cut falls between 183,000 and 194,000 UTF-8 bytes of the message, whatever the
# text is (English prose, Python source, accented Portuguese, digit tables). It
# is a byte limit, not a token limit: 168,000 characters of digit tables
# (177,000 input tokens) arrive whole, while 334,000 characters of prose
# (about 85,000 tokens) arrive cut at about 186,000.
#
# What the conversation itself can hold is far larger: four messages of
# 139,000 characters each in one conversation (152,000 input tokens) were all
# recalled. So a long input is delivered as several messages (multipart, see
# split_prompt_parts) instead of being shortened.

# Multipart delivery on the persistent worker. One message never exceeds this
# many UTF-8 bytes, frame included. 0 or negative turns multipart off, which
# restores the single-message behaviour and its budgets below.
_PART_MAX_BYTES = _env_int("HERMES_AGY_PART_MAX_BYTES", 140_000)
# Bytes reserved for the frame frame_prompt_part() puts around a part.
_PART_FRAME_BYTES = 900
# With multipart on, the whole input may be this large before its middle is
# elided (about 200,000 tokens of prose, under the 239,000 the plugin declares).
_WORKER_PROMPT_MAX_CHARS = _env_int("HERMES_AGY_WORKER_PROMPT_MAX_CHARS", 800_000)
# With multipart on, one tool result may be this large before it is cut.
_WORKER_TOOL_RESULT_MAX_CHARS = _env_int("HERMES_AGY_TOOL_RESULT_MAX_CHARS", 400_000)

# Single-message budget (multipart off, and the oneshot path, which has no
# conversation to deliver parts into): a hard character budget with MIDDLE
# elision, so the head (system preamble + tool schemas) and the tail (latest
# user request, tool-loop objective) survive agy's cut.
#
# Override with HERMES_AGY_PROMPT_MAX_CHARS; 0 or negative disables the budget.
_PROMPT_MAX_CHARS = _env_int("HERMES_AGY_PROMPT_MAX_CHARS", 120_000)
# Single-message cap of one tool result.
_TOOL_RESULT_MAX_CHARS = 20_000

# A request restated next to the transcript (objective and tool-loop anchors)
# is bounded: the full text is already in the conversation, and restating a
# long request on every tool turn would spend the context on copies of it.
_RESTATEMENT_MAX_CHARS = 4_000
_RESTATEMENT_HEAD_CHARS = 3_000
_RESTATEMENT_TAIL_CHARS = 600

# Sections that must never be elided, even when over budget: instruction blocks,
# the tool schemas (cut in their middle they are invalid JSON) and the newest
# request the model has to answer (the last section is always kept as well,
# whatever it is).
_PROMPT_PINNED_PREFIXES = ("### ", "Available tools")
_TRANSCRIPT_SECTION_PREFIX = "Conversation transcript:"

# agy attaches media pixels through its own native read step: the model must
# call read_file/view_file on the referenced path, and the CLI then inlines the
# file as a media block. The blanket "native tools are disabled" preamble would
# otherwise forbid the one tool call the media pipeline requires (verified in
# session transcripts: reference present + read denied -> no pixels -> "no
# puedo verlo"). This section is injected ONLY on turns whose assembled prompt
# actually carries a media reference.
_MEDIA_VIEWING_SECTION = (
    "### MEDIA VIEWING PROTOCOL (applies ONLY to this turn's media references):\n"
    "Messages may contain markdown media references: [Image](/abs/path), [Video](/abs/path) or [Audio](/abs/path).\n"
    "These are real attachments, not prose. To see the media, call your native read_file (or view_file) "
    "tool on that exact path once — the runtime then attaches the file's pixels to your context, and you "
    "answer from what you see.\n"
    "This read is the ONLY permitted use of a native tool in this mode; every other native tool stays "
    "disabled. Never describe media you have not loaded this way, and never narrate loading steps — "
    "just read the file, then answer."
)


def _apply_media_viewing_protocol(sections: list[str]) -> list[str]:
    """Insert the media viewing section when the assembled prompt carries a reference."""
    if not any(_MEDIA_REF_RE.search(section) for section in sections):
        return sections
    insert_at = 0
    for index, section in enumerate(sections):
        if section.startswith("Conversation transcript:"):
            insert_at = index
            break
    sections.insert(insert_at, _MEDIA_VIEWING_SECTION)
    return sections


_TOOL_CALL_PREFIXES = tuple(
    "<tool_call>"[:i] for i in range(len("<tool_call>"), 0, -1)
)

_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"


def _find_tool_call_close(text: str, start: int = 0) -> int:
    """Index of the first ``</tool_call>`` outside a JSON string, or -1.

    A tool argument may legitimately contain the literal closing tag (for
    example a patch or a file whose content includes protocol markup). A plain
    ``str.find`` truncates the call at that literal and hands invalid JSON to
    the parser. This scanner tracks JSON double-quoted strings and backslash
    escapes, so only a tag at string depth zero closes the frame. It is the
    single frame boundary used by both the incremental stream splitter and
    :func:`_parse_tool_block`.
    """
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if text.startswith(_TOOL_CALL_CLOSE, index):
            return index
    return -1


def _render_message_content(content: Any, media_sink: Optional[set] = None) -> str:
    """Normalize multimodal or structured message content into a string.

    Synthetic media notes are translated to canonical agy references so the
    CLI's media pipeline can attach the real pixels (see _normalize_media_notes).
    Refused media (unknown extension, non-public host, non-media type) is
    dropped entirely: nothing unverifiable is ever handed to agy.
    """
    rendered: str
    if content is None:
        return ""
    if isinstance(content, dict):
        if "text" in content:
            rendered = _clean_rendered_text(str(content.get("text") or ""))
        else:
            rendered = _clean_rendered_text(str(content.get("content") or ""))
        return _normalize_media_notes(rendered, media_sink)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                cleaned = _clean_rendered_text(item)
                if cleaned:
                    parts.append(cleaned)
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    cleaned = _clean_rendered_text(str(item.get("text") or ""))
                    if cleaned:
                        parts.append(cleaned)
                elif item_type in ("image_url", "image", "video_url", "video"):
                    img_url_obj = (
                        item.get(item_type)
                        if item_type in item
                        else (
                            item.get("image_url")
                            or item.get("image")
                            or item.get("video_url")
                            or item.get("video")
                        )
                    )
                    if isinstance(img_url_obj, dict):
                        url_val = img_url_obj.get("url", "")
                    else:
                        url_val = str(img_url_obj or "")
                    if not url_val and isinstance(item.get("url"), str):
                        url_val = item["url"]
                    if url_val:
                        fpath, mtype = _materialize_media(url_val, media_sink)
                        if fpath:
                            parts.append(f"[{mtype}]({fpath})")
                        else:
                            # Fail-closed: a refused or unmaterializable
                            # reference must NOT degrade into a raw URL /
                            # path handed to agy as pseudo-media.
                            logger.debug(
                                "Dropped %s part (materialization refused): %.120s",
                                item_type,
                                url_val,
                            )
                elif item_type == "input_audio":
                    parts.append("[Audio input]")
                elif "text" in item:
                    cleaned = _clean_rendered_text(str(item.get("text") or ""))
                    if cleaned:
                        parts.append(cleaned)
        return _normalize_media_notes("\n".join(p for p in parts if p).strip(), media_sink)
    return _normalize_media_notes(_clean_rendered_text(str(content)), media_sink)


def _bounded_restatement(user_text: str) -> str:
    """A request as it is restated beside the transcript: whole, or its two ends.

    Only for the anchors that repeat a request the conversation already holds
    in full. The section that delivers the request itself is never bounded.
    """
    if len(user_text) <= _RESTATEMENT_MAX_CHARS:
        return user_text
    omitted = len(user_text) - _RESTATEMENT_HEAD_CHARS - _RESTATEMENT_TAIL_CHARS
    return (
        user_text[:_RESTATEMENT_HEAD_CHARS]
        + f"\n[... {omitted} chars of this request are not repeated here; "
        "its full text is the User message in the conversation ...]\n"
        + user_text[-_RESTATEMENT_TAIL_CHARS:]
    )


def _cap_tool_output(rendered_content: str, cap: int) -> str:
    """Cut one tool result past *cap* characters, keeping its head and its tail."""
    if cap <= 0 or len(rendered_content) <= cap:
        return rendered_content
    return (
        rendered_content[: int(cap * 0.6)]
        + f"\n[... large tool output truncated: {len(rendered_content)} chars ...]\n"
        + rendered_content[-int(cap * 0.15):]
    )


def _latest_user_request_section(user_text: str) -> str:
    """Format the canonical LATEST USER REQUEST section and anti-loop instruction."""
    return (
        f"### LATEST USER REQUEST TO ANSWER:\nUser:\n{user_text}\n\n"
        "INSTRUCTION: Respond directly and specifically to the LATEST USER REQUEST above. "
        "Do NOT repeat previous architectural summaries, code reviews, or overview boilerplate unless explicitly asked."
        " If the LATEST USER REQUEST comments on, questions, or gives feedback about prior work rather than asking to continue it, address THAT message and do NOT silently continue the earlier task. If the user's intent is genuinely ambiguous, ask ONE short clarifying question instead of proceeding."
    )


# Core strips private message keys before providers see the request. Match only
# the complete known recovery text, never arbitrary user text containing it.
_EMPTY_TOOL_RESPONSE_NUDGE = (
    "You just executed tool calls but returned an empty response. "
    "Please process the tool results above and continue with the task."
)


def _is_recovery_nudge(message: dict[str, Any]) -> bool:
    return bool(message.get("_empty_recovery_synthetic")) or (
        _render_message_content(message.get("content")) == _EMPTY_TOOL_RESPONSE_NUDGE
    )


def _latest_user_text(messages: Sequence[Any]) -> str:
    """Return the rendered text of the last user message, or an empty string."""
    for msg in reversed(list(messages or [])):
        if isinstance(msg, dict) and str(msg.get("role") or "").strip().lower() == "user":
            if not _is_recovery_nudge(msg):
                return _render_message_content(msg.get("content"))
    return ""


def _pending_user_request_section(user_text: str) -> str:
    """Restate the user request that the current tool loop is serving.

    Tool-result turns otherwise end with a generic "continue" line, so after a
    few tool calls the request falls far behind the newest text the model sees
    and the model drifts back to the dominant pattern in the transcript (for
    example re-sending an earlier completion report).
    """
    return (
        f"### USER REQUEST THIS TOOL LOOP IS SERVING:\nUser:\n{_bounded_restatement(user_text)}\n\n"
        "INSTRUCTION: Use the tool results only to answer THIS request. "
        "When you have enough information, answer it directly and specifically. "
        "Do NOT repeat earlier reports or summaries from the transcript unless this request asks for them."
    )


def _tool_loop_objective_section(user_text: str) -> str:
    """Anchor the request a tool loop serves above the transcript.

    A full prompt that ends on a tool result (first call after a worker
    respawn or a broken pipe) otherwise names the request only at the tail,
    and a long transcript drowns a tail-only instruction.
    """
    return (
        "### ACTIVE OBJECTIVE:\n"
        f"The user request being served is:\n\"{_bounded_restatement(user_text)}\"\n\n"
        "CRITICAL: The conversation transcript below contains prior history and the tool calls made so far. "
        "Use the tool results ONLY to address the ACTIVE OBJECTIVE above. "
        "When tool execution is complete, your final answer must directly address this objective. "
        "Do NOT repeat earlier completion reports, summaries, or boilerplate from previous tasks, "
        "and do NOT resume an earlier task."
    )


def _elide_blocks(
    blocks: list[str], room: int, head_ratio: float = 0.55
) -> tuple[list[str], int, int, int]:
    """Keep the head and the tail of a block list, dropping the middle.

    Returns ``(kept_blocks, head_count, dropped_count, dropped_chars)``.
    ``head_count`` is how many of the returned blocks came from the head, so the
    caller can put the elision marker exactly at the cut. ``room`` is the
    character budget for the kept blocks, separators included.
    """
    if room <= 0 or not blocks:
        return [], 0, len(blocks), sum(len(b) for b in blocks)
    if len(blocks) <= 2:
        return list(blocks), len(blocks), 0, 0

    head_budget = int(room * head_ratio)
    tail_budget = room - head_budget
    head: list[str] = []
    tail: list[str] = []
    head_chars = 0
    tail_chars = 0

    for block in blocks:
        cost = len(block) + 2
        if head_chars + cost > head_budget:
            break
        head.append(block)
        head_chars += cost

    for block in reversed(blocks[len(head):]):
        cost = len(block) + 2
        if tail_chars + cost > tail_budget:
            break
        tail.insert(0, block)
        tail_chars += cost

    dropped = blocks[len(head):len(blocks) - len(tail)]
    return [*head, *tail], len(head), len(dropped), sum(len(b) for b in dropped)


def _fit_prompt_sections(sections: list[str], limit: int | None = None) -> list[str]:
    """Keep the assembled prompt under a character budget by eliding the middle.

    Mirrors the cursor prompt budget in agentes-chat: the head (system preamble,
    tool schemas) and the tail (latest user request, tool-loop instructions)
    survive; the middle of the transcript is dropped with a marker. agy truncates
    from the tail when a prompt is too large, which silently deletes the request
    the model has to answer (issue #29). Pinned sections (``### ...``) and the
    last section are never elided.
    """
    budget = _PROMPT_MAX_CHARS if limit is None else limit
    if budget <= 0:
        return sections

    def _size(items: list[str]) -> int:
        return sum(len(s) for s in items) + 2 * max(len(items) - 1, 0)

    total = _size(sections)
    if total <= budget:
        return sections

    result = list(sections)

    # 1) Elide the middle of the transcript first: it is the largest block and
    #    the least critical one to keep whole.
    idx = next(
        (i for i, s in enumerate(result) if s.startswith(_TRANSCRIPT_SECTION_PREFIX)),
        None,
    )
    if idx is not None:
        prefix = _TRANSCRIPT_SECTION_PREFIX
        body = result[idx][len(prefix):]
        blocks = [b for b in body.split("\n\n") if b.strip()]
        overhead = total - len(result[idx])
        room = budget - overhead
        kept, head_count, dropped, dropped_chars = _elide_blocks(blocks, room)
        if dropped:
            marker = (
                f"[... {dropped} transcript blocks elided "
                f"({dropped_chars} chars) to fit the prompt budget ...]"
            )
            kept.insert(head_count, marker)
            result[idx] = prefix + "\n\n" + "\n\n".join(kept)
            total = _size(result)

    # 2) Still over budget: trim the middle of oversized non-pinned sections
    #    (tool schemas), never the last one.
    if total > budget:
        for i in range(len(result) - 1):
            if total <= budget:
                break
            section = result[i]
            if section.startswith(_PROMPT_PINNED_PREFIXES):
                continue
            excess = total - budget
            keep = len(section) - excess - 400
            if keep < 2000:
                continue
            head_len = int(keep * 0.6)
            tail_len = keep - head_len
            trimmed = (
                section[:head_len]
                + f"\n\n[... {len(section) - keep} chars elided from the middle of this section to fit the prompt budget ...]\n\n"
                + section[-tail_len:]
            )
            total -= len(section) - len(trimmed)
            result[i] = trimmed

    return result


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    media_sink: Optional[set] = None,
    multipart: bool = False,
) -> str:
    """Assemble Hermes message history, system prompt, and tool schemas into a canonical prompt.

    ``multipart``: the caller delivers the result as several messages
    (``split_prompt_parts``), so nothing is shortened to fit one message: tool
    results keep their content up to ``_WORKER_TOOL_RESULT_MAX_CHARS``, older
    tool results are pruned only when the history is larger than the worker
    limit, and the middle elision starts at ``_WORKER_PROMPT_MAX_CHARS``.
    """
    try:
        from agent.acp_openai_bridge import render_tool_bridge_sections
        tool_sections = render_tool_bridge_sections(tools, tool_choice)
    except Exception:
        tool_sections = []
        if tools:
            tool_sections.append(
                "Available tools (OpenAI function schema):\n"
                + json.dumps(tools, ensure_ascii=False)
            )

    sections: list[str] = [*_PROMPT_PREAMBLE, *tool_sections]

    valid_messages = [m for m in messages if isinstance(m, dict)]
    last_msg = valid_messages[-1] if valid_messages else None
    last_role = str(last_msg.get("role") or "").strip().lower() if last_msg else ""

    system_parts: list[str] = []
    history_messages: list[dict[str, Any]] = []

    for msg in valid_messages:
        role = str(msg.get("role") or "unknown").strip().lower()
        if role == "system":
            rendered = _render_message_content(msg.get("content"), media_sink)
            if rendered:
                system_parts.append(rendered)
        else:
            history_messages.append(msg)

    if system_parts:
        sections.append("### SYSTEM INSTRUCTIONS (HERMES AGENT):\n" + "\n\n".join(f"System:\n{p}" for p in system_parts))

    _HISTORICAL_TOOL_RETENTION_COUNT = 8
    _HISTORICAL_TOOL_MAX_CHARS = 300
    _TOOL_MAX_CHARS_CAP = _WORKER_TOOL_RESULT_MAX_CHARS if multipart else _TOOL_RESULT_MAX_CHARS

    transcript: list[str] = []
    recovery_turn = last_role == "user" and last_msg is not None and _is_recovery_nudge(last_msg)
    prior_messages = history_messages[:-1] if last_role == "user" and not recovery_turn and len(history_messages) > 1 else history_messages

    tool_indices = [
        i for i, msg in enumerate(prior_messages)
        if str(msg.get("role") or "").strip().lower() == "tool"
    ]
    recent_tool_cutoff = (
        set(tool_indices[-_HISTORICAL_TOOL_RETENTION_COUNT:])
        if len(tool_indices) > _HISTORICAL_TOOL_RETENTION_COUNT
        else set(tool_indices)
    )
    if multipart and _WORKER_PROMPT_MAX_CHARS > 0:
        # Delivered in parts, the history has room: older tool results are
        # reduced to a stub only when keeping them whole would not fit.
        history_chars = sum(len(str(msg.get("content") or "")) for msg in prior_messages)
        if history_chars <= _WORKER_PROMPT_MAX_CHARS:
            recent_tool_cutoff = set(tool_indices)

    for i, message in enumerate(prior_messages):
        role = str(message.get("role") or "unknown").strip().lower()
        rendered_content = _render_message_content(message.get("content"))

        if role == "tool":
            tool_id = str(message.get("tool_call_id") or message.get("name") or "tool").strip()
            if i not in recent_tool_cutoff and len(rendered_content) > _HISTORICAL_TOOL_MAX_CHARS:
                rendered_content = (
                    rendered_content[:200]
                    + f"\n[... tool output truncated: {len(rendered_content)} chars ...]\n"
                    + rendered_content[-100:]
                )
            else:
                rendered_content = _cap_tool_output(rendered_content, _TOOL_MAX_CHARS_CAP)
            transcript.append(f"Tool Result ({tool_id}):\n{rendered_content}")
            continue

        if role == "assistant":
            parts = []
            if "Tool Result (" in rendered_content:
                rendered_content = rendered_content.split("Tool Result (")[0].strip()
            if rendered_content:
                parts.append(rendered_content)
            if tool_calls := message.get("tool_calls"):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function") or {}
                        call_obj = {
                            "id": tc.get("id") or "call_1",
                            "type": "function",
                            "function": {
                                "name": fn.get("name", ""),
                                "arguments": fn.get("arguments", "{}") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments", {}), ensure_ascii=False)
                            }
                        }
                        parts.append(f"<tool_call>{json.dumps(call_obj, ensure_ascii=False)}</tool_call>")
            if parts:
                transcript.append("Assistant:\n" + "\n".join(parts))
            continue

        label = _ROLE_LABELS.get(role, "Context")
        if rendered_content:
            transcript.append(f"{label}:\n{rendered_content}")

    if recovery_turn:
        objective = _latest_user_text(history_messages)
        if objective:
            sections.append(_tool_loop_objective_section(objective))
    elif last_role == "user" and last_msg is not None and len(history_messages) > 1:
        user_text = _render_message_content(last_msg.get("content"), media_sink)
        sections.append(
            "### ACTIVE OBJECTIVE:\n"
            f"The user has submitted a new request:\n\"{_bounded_restatement(user_text)}\"\n\n"
            "CRITICAL: The conversation transcript below is historical background. "
            "Do NOT silently continue prior tool execution loops or earlier coding tasks unless explicitly requested by the ACTIVE OBJECTIVE. "
            "Focus your attention on answering the new request."
        )
    elif last_role == "tool":
        pending_objective = _latest_user_text(history_messages)
        if pending_objective:
            sections.append(_tool_loop_objective_section(pending_objective))

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    if last_role == "tool":
        sections.append(
            "### LATEST TOOL RESULTS RECEIVED.\n"
            "INSTRUCTION: Evaluate the latest tool results in the transcript above and continue the task. "
            "If more tools are needed, emit <tool_call> tags. If you have enough information, "
            "respond clearly to the user without repeating prior summaries."
        )
        pending = _latest_user_text(history_messages)
        if pending:
            sections.append(_pending_user_request_section(pending))
    elif last_role == "user" and last_msg is not None:
        user_text = _latest_user_text(history_messages)
        if user_text:
            sections.append(_latest_user_request_section(user_text))
    else:
        sections.append("Continue the conversation from the latest message.")
    assembled = [s.strip() for s in _apply_media_viewing_protocol(sections) if s and s.strip()]
    return "\n\n".join(
        _fit_prompt_sections(assembled, limit=_WORKER_PROMPT_MAX_CHARS if multipart else None)
    )


def _messages_match_prefix(history: Sequence[dict[str, Any]], incoming: Sequence[dict[str, Any]]) -> bool:
    """Return True if incoming messages strictly extend history as a continuation."""
    if not history or len(incoming) <= len(history):
        return False
    for i, h_msg in enumerate(history):
        inc_msg = incoming[i]
        if not isinstance(inc_msg, dict) or not isinstance(h_msg, dict):
            return False
        if inc_msg.get("role") != h_msg.get("role"):
            return False
        if inc_msg.get("content") != h_msg.get("content"):
            return False
        if inc_msg.get("tool_calls") != h_msg.get("tool_calls"):
            return False
    return True


def _format_delta_prompt(
    new_messages: Sequence[dict[str, Any]],
    pending_user_text: str = "",
    media_sink: Optional[set] = None,
    multipart: bool = False,
) -> str:
    """Format only the incremental messages in an ongoing multi-turn interaction.

    ``pending_user_text`` is the latest user request from the full history.
    A tool-result delta does not contain it, so the caller passes it in and it
    is restated after the continue instruction.
    """
    valid_messages = [m for m in new_messages if isinstance(m, dict)]
    if not valid_messages:
        return ""

    last_msg = valid_messages[-1]
    last_role = str(last_msg.get("role") or "").strip().lower()

    recovery_turn = last_role == "user" and _is_recovery_nudge(last_msg)
    if last_role == "user" and not recovery_turn:
        prior_messages = valid_messages[:-1]
    else:
        prior_messages = valid_messages

    parts: list[str] = []
    for msg in prior_messages:
        role = str(msg.get("role") or "").strip().lower()
        rendered_content = _render_message_content(msg.get("content"), media_sink)
        if role == "tool":
            tool_id = str(msg.get("tool_call_id") or msg.get("name") or "tool").strip()
            rendered_content = _cap_tool_output(
                rendered_content,
                _WORKER_TOOL_RESULT_MAX_CHARS if multipart else _TOOL_RESULT_MAX_CHARS,
            )
            parts.append(f"Tool Result ({tool_id}):\n{rendered_content}")
            continue
        if role == "assistant":
            subparts = []
            if "Tool Result (" in rendered_content:
                rendered_content = rendered_content.split("Tool Result (")[0].strip()
            if rendered_content:
                subparts.append(rendered_content)
            if tool_calls := msg.get("tool_calls"):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function") or {}
                        call_obj = {
                            "id": tc.get("id") or "call_1",
                            "type": "function",
                            "function": {
                                "name": fn.get("name", ""),
                                "arguments": fn.get("arguments", "{}") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments", {}), ensure_ascii=False)
                            }
                        }
                        subparts.append(f"<tool_call>{json.dumps(call_obj, ensure_ascii=False)}</tool_call>")
            if subparts:
                parts.append("Assistant:\n" + "\n".join(subparts))
            continue
        label = _ROLE_LABELS.get(role, "Context")
        if rendered_content:
            parts.append(f"{label}:\n{rendered_content}")

    if last_role == "tool":
        parts.append(
            "Continue the conversation from the latest tool result. "
            "REMINDER: You are in HEADLESS INFERENCE MODE. Native tools are disabled. "
            "If you need to call a tool, emit ONLY <tool_call>{...}</tool_call> blocks in your text output. "
            "Do NOT invoke native tools directly."
        )
        if pending_user_text:
            parts.append(_pending_user_request_section(pending_user_text))
    elif last_role == "user":
        user_text = (pending_user_text or _latest_user_text(valid_messages)) if recovery_turn else _render_message_content(last_msg.get("content"), media_sink)
        if user_text:
            parts.append(_latest_user_request_section(user_text))
    else:
        parts.append("Continue the conversation from the latest message above.")
    assembled = [s.strip() for s in _apply_media_viewing_protocol(parts) if s and s.strip()]
    return "\n\n".join(
        _fit_prompt_sections(assembled, limit=_WORKER_PROMPT_MAX_CHARS if multipart else None)
    )


# --- Multipart delivery -------------------------------------------------------

# What a part that is not the last one asks the model to reply.
MULTIPART_ACK = "ACK"


def multipart_enabled() -> bool:
    """Whether a long input is delivered as several messages (read at call time)."""
    return _PART_MAX_BYTES > _PART_FRAME_BYTES


def part_payload_limit() -> int:
    """UTF-8 bytes of content one part may carry, its frame left out."""
    return _PART_MAX_BYTES - _PART_FRAME_BYTES


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _split_oversize(block: str, limit: int) -> list[str]:
    """Pieces of a block larger than *limit* bytes: whole lines, a longer line by bytes."""
    pieces: list[str] = []
    for line in block.splitlines(keepends=True):
        data = line.encode("utf-8")
        if len(data) <= limit:
            pieces.append(line)
            continue
        start = 0
        while start < len(data):
            end = min(start + limit, len(data))
            # Never cut inside a multi-byte character.
            while end < len(data) and (data[end] & 0xC0) == 0x80:
                end -= 1
            if end <= start:  # a limit smaller than one character: take that character
                end = start + 1
                while end < len(data) and (data[end] & 0xC0) == 0x80:
                    end += 1
            pieces.append(data[start:end].decode("utf-8"))
            start = end
    return pieces


def split_prompt_parts(text: str, max_bytes: int | None = None) -> list[str]:
    """Split *text* into consecutive parts of at most *max_bytes* UTF-8 bytes.

    Nothing is dropped or rewritten: joining the parts gives *text* back. Cuts
    fall on blank lines where possible, then on line ends, and inside a line
    only when that single line is larger than a part. The last part is as full
    as the limit allows. One part when the text fits, or when *max_bytes* is
    not positive.
    """
    limit = part_payload_limit() if max_bytes is None else max_bytes
    if limit <= 0 or len(text) * 4 <= limit or _utf8_len(text) <= limit:
        return [text]
    pieces: list[tuple[str, int]] = []
    for block in re.findall(r".+?(?:\n{2,}|\Z)", text, flags=re.S):
        size = _utf8_len(block)
        if size <= limit:
            pieces.append((block, size))
        else:
            pieces.extend((piece, _utf8_len(piece)) for piece in _split_oversize(block, limit))

    # The last part is the one the model acts on, so it is filled from the END:
    # the request and the newest context travel together with the instruction,
    # instead of a few leftover lines arriving alone.
    tail_start = len(pieces)
    tail_bytes = 0
    while tail_start > 0 and tail_bytes + pieces[tail_start - 1][1] <= limit:
        tail_start -= 1
        tail_bytes += pieces[tail_start][1]

    parts: list[str] = []
    current: list[str] = []
    current_bytes = 0
    for piece, size in pieces[:tail_start]:
        if current and current_bytes + size > limit:
            parts.append("".join(current))
            current, current_bytes = [], 0
        current.append(piece)
        current_bytes += size
    if current:
        parts.append("".join(current))
    parts.append("".join(piece for piece, _ in pieces[tail_start:]))
    return parts


def frame_prompt_part(content: str, index: int, total: int) -> str:
    """Wrap part *index* (1-based) of *total* for delivery as its own message.

    A part that is not the last asks for nothing but an acknowledgement, so the
    model does not start on half an input. The last part tells the model the
    input is complete and where the rest of it is.
    """
    if total <= 1:
        return content
    if index < total:
        head = (
            f"[MULTIPART INPUT {index}/{total}]\n"
            f"This is part {index} of {total} of ONE long message, delivered in several parts "
            "because of a transport size limit. Do not act on it yet: do not answer it, do not "
            "summarize it and do not call any tool. More parts follow. "
            f"Reply with exactly: {MULTIPART_ACK}"
        )
    else:
        head = (
            f"[MULTIPART INPUT {total}/{total}: FINAL PART]\n"
            f"The long message is now complete. Its parts 1 to {total - 1} are the previous "
            "messages of this conversation marked MULTIPART INPUT. Read all parts in order as one "
            "continuous message, then follow the instructions it contains."
        )
    return (
        f"{head}\n--- BEGIN PART {index}/{total} ---\n"
        f"{content}\n--- END PART {index}/{total} ---"
    )


def media_reference_paths(messages: Any, media_sink: Optional[set] = None) -> set[str]:
    """Every media path the rendered messages reference: ``[Image](/abs/path)``.

    Shared by the stream's native-tool allowlist and the client's worker
    permissions stamp, so both describe the same media contract. Rendering
    materializes references (downloads, cache files) into *media_sink* when
    given, so the caller can clean them up at session close.
    """
    paths: set[str] = set()
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        rendered = _render_message_content(message.get("content"), media_sink)
        for path in _MEDIA_REF_RE.findall(rendered):
            if path:
                paths.add(os.path.normpath(path))
    return paths


def _longest_tool_call_prefix_match(text: str) -> int:
    """Return the length of the longest suffix of `text` that matches a prefix of `<tool_call>`."""
    for prefix in _TOOL_CALL_PREFIXES:
        if text.endswith(prefix):
            return len(prefix)
    return 0


def _has_unfinished_tool_call_frame(text: str) -> bool:
    """True when *text* ends inside a ``<tool_call>`` frame.

    Uses the incremental splitter's frame boundary (:func:`_find_tool_call_close`)
    so a complete block whose JSON contains a literal closing tag is not
    mistaken for a truncated one. An opening tag without a matching close, or
    a trailing fragment of the opening tag (``<tool_cal``), is unfinished.
    """
    cursor = 0
    while True:
        start = text.find(_TOOL_CALL_OPEN, cursor)
        if start == -1:
            # A lone "<" is ordinary text (a comparison operator, an unfinished
            # HTML tag, ...), not evidence of a truncated <tool_call> frame.
            # Two or more matching characters ("<t" ... "<tool_call") remain
            # fail-closed: the shortest useful proof of protocol framing.
            return _longest_tool_call_prefix_match(text) > 1
        close = _find_tool_call_close(text, start + len(_TOOL_CALL_OPEN))
        if close == -1:
            return True
        cursor = close + len(_TOOL_CALL_CLOSE)


def _tool_call_from_object(obj: Any, ordinal: int) -> Any | None:
    """Build one tool-call delta object from a decoded JSON value.

    Mirrors the core ACP parser: the value must be an object whose
    ``function`` is an object with a non-blank string name. Anything else is
    malformed and returns None, so a nameless call is never emitted.
    """
    if not isinstance(obj, dict):
        return None
    fn = obj.get("function")
    if not isinstance(fn, dict):
        return None
    fn_name = fn.get("name")
    if not isinstance(fn_name, str) or not fn_name.strip():
        return None
    fn_name = fn_name.strip()
    call_id = obj.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        call_id = f"call_{ordinal}"
    fn_args = fn.get("arguments", "{}")
    if not isinstance(fn_args, str):
        fn_args = json.dumps(fn_args, ensure_ascii=False)
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=fn_name, arguments=fn_args),
    )


def _has_unparseable_tool_call_frame(text: str) -> bool:
    """True when *text* keeps a closed ``<tool_call>`` frame after parsing.

    :func:`_parse_tool_block` consumes every frame it can turn into a call and
    leaves the unparseable ones in its cleaned text: invalid JSON, a missing
    or blank ``function`` name, a non-object value, or an empty body. A
    closed frame surviving in that cleaned text is therefore an explicit
    protocol frame with no call, which must fail the turn instead of being
    published as assistant text. Unclosed frames are
    :func:`_has_unfinished_tool_call_frame`'s concern.
    """
    _calls, cleaned = _parse_tool_block(text)
    start = cleaned.find(_TOOL_CALL_OPEN)
    return (
        start != -1
        and _find_tool_call_close(cleaned, start + len(_TOOL_CALL_OPEN)) != -1
    )


def _extract_tagged_tool_calls(block: str) -> tuple[list[Any], str]:
    """Extract ``<tool_call>{...}</tool_call>`` frames using the shared scanner.

    The frame boundary is :func:`_find_tool_call_close`, so a literal closing
    tag inside a JSON string does not terminate the block. Text outside valid
    frames, and a frame whose body is not a valid call, are returned as
    ``cleaned_text``.
    """
    calls: list[Any] = []
    text_parts: list[str] = []
    cursor = 0
    while True:
        start = block.find(_TOOL_CALL_OPEN, cursor)
        if start == -1:
            text_parts.append(block[cursor:])
            break
        text_parts.append(block[cursor:start])
        close = _find_tool_call_close(block, start + len(_TOOL_CALL_OPEN))
        if close == -1:
            text_parts.append(block[start:])
            break
        end = close + len(_TOOL_CALL_CLOSE)
        body = block[start + len(_TOOL_CALL_OPEN):close].strip()
        parsed = None
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = None
        call = _tool_call_from_object(parsed, len(calls) + 1)
        if call is None:
            text_parts.append(block[start:end])
        else:
            calls.append(call)
        cursor = end
    if not calls:
        # No frame was consumed: hand the input back untouched. Joining the
        # pieces would insert separators at every tag boundary and corrupt
        # ordinary prose that merely mentions the markup.
        return [], block.strip()
    return calls, "\n".join(part.strip() for part in text_parts if part and part.strip()).strip()


def _parse_tool_block(block: str) -> tuple[list[Any], str]:
    """Parse tool call blocks, returning (tool_call_deltas, cleaned_text)."""
    calls: list[Any] = []
    cleaned_text = ""
    if _TOOL_CALL_OPEN in block:
        calls, cleaned_text = _extract_tagged_tool_calls(block)
    else:
        try:
            from agent.acp_openai_bridge import extract_tool_calls_from_text
            calls, cleaned_text = extract_tool_calls_from_text(block)
        except Exception:
            cleaned_text = block

    deltas = []
    for i, call in enumerate(calls):
        fn = getattr(call, "function", None)
        name = getattr(fn, "name", "") if fn else ""
        args = getattr(fn, "arguments", "{}") if fn else "{}"
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        deltas.append(
            SimpleNamespace(
                index=i,
                id=getattr(call, "id", f"call_{i+1}"),
                type="function",
                function=SimpleNamespace(name=name, arguments=args),
            )
        )
    return deltas, cleaned_text.strip()
