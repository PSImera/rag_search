from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import YOUTUBE_ROOT

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url_or_id: str) -> str:
    """Достаёт video_id из youtu.be/<id>, youtube.com/watch?v=, youtube.com/live/<id> или голого id.

    Мусорные параметры (?si=...) отбрасываются.
    """
    s = url_or_id.strip()
    if _ID_RE.match(s):
        return s
    m = re.search(r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|live/|shorts/|embed/))([A-Za-z0-9_-]{11})", s)
    if m:
        return m.group(1)
    raise ValueError(f"Не удалось извлечь video_id из: {url_or_id}")


def _sanitize(name: str, max_len: int = 80) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip().rstrip(".")
    name = re.sub(r"\s+", " ", name)
    return name[:max_len].strip()


def video_dir(video_id: str, channel: str, title: str) -> Path:
    """Человекочитаемая папка кэша: data/youtube/<канал> - <название> [id]/"""
    label = _sanitize(f"{channel} - {title}") or "unknown"
    d = YOUTUBE_ROOT / f"{label} [{video_id}]"
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_video_dir(video_id: str) -> Path | None:
    """Ищет уже существующую папку кэша по [video_id] в имени."""
    if not YOUTUBE_ROOT.exists():
        return None
    for d in YOUTUBE_ROOT.iterdir():
        if d.is_dir() and d.name.endswith(f"[{video_id}]"):
            return d
    return None


def fetch_video_info(video_id: str) -> dict[str, Any]:
    """Метаданные видео через yt-dlp (без скачивания). Кэшируется в info.json."""
    existing = find_video_dir(video_id)
    if existing and (existing / "info.json").exists():
        return json.loads((existing / "info.json").read_text(encoding="utf-8"))

    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "extract_flat": False}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    slim = {
        "video_id": video_id,
        "title": info.get("title") or "",
        "channel": info.get("channel") or info.get("uploader") or "",
        "duration": info.get("duration"),
        "is_live": bool(info.get("is_live")),
        "was_live": bool(info.get("was_live")),
    }
    d = video_dir(video_id, slim["channel"], slim["title"])
    (d / "info.json").write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    return slim


def fetch_comments(video_id: str, info: dict[str, Any]) -> list[dict[str, Any]]:
    """Комментарии через yt-dlp (--write-comments эквивалент). Кэш: comments.json.

    Возвращает нормализованный список; пустой список — если комментарии отключены/недоступны.
    """
    d = video_dir(video_id, info["channel"], info["title"])
    cache = d / "comments.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    import yt_dlp

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "getcomments": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            raw = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        comments = raw.get("comments") or []
    except Exception as exc:
        print(f"[warn] Комментарии для {video_id} недоступны: {exc}")
        comments = []

    normalized: list[dict[str, Any]] = []
    for c in comments:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        ts = c.get("timestamp")
        date = (
            datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            if ts
            else ""
        )
        parent = c.get("parent") or "root"
        normalized.append(
            {
                "comment_id": str(c.get("id")),
                "parent_id": None if parent == "root" else str(parent),
                "author": c.get("author") or "",
                "author_id": c.get("author_id") or "",
                "date": date,
                "likes": int(c.get("like_count") or 0),
                "text": text,
            }
        )
    cache.write_text(json.dumps(normalized, ensure_ascii=False), encoding="utf-8")
    return normalized


def fetch_transcript(video_id: str, info: dict[str, Any], lang: str = "ru") -> list[dict[str, Any]] | None:
    """Транскрипция: youtube-transcript-api (ручные + автосубтитры, фолбэк en),
    запасной путь — yt-dlp --write-auto-subs (vtt). Кэш: transcript_<lang>.json.

    Возвращает список {text, start, duration} или None, если транскрипции нет.
    """
    d = video_dir(video_id, info["channel"], info["title"])
    for cached_lang in (lang, "en"):
        cache = d / f"transcript_{cached_lang}.json"
        if cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
            return data if data else None

    snippets = _transcript_via_api(video_id, lang)
    used_lang = lang
    if snippets is None:
        snippets = _transcript_via_ytdlp(video_id, lang)
    if snippets is None and lang != "en":
        snippets = _transcript_via_api(video_id, "en")
        used_lang = "en"

    cache = d / f"transcript_{used_lang if snippets else lang}.json"
    cache.write_text(json.dumps(snippets or [], ensure_ascii=False), encoding="utf-8")
    return snippets


def _transcript_via_api(video_id: str, lang: str) -> list[dict[str, Any]] | None:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=[lang])
        return [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]
    except Exception as exc:
        print(f"[info] youtube-transcript-api ({lang}) не сработал для {video_id}: {type(exc).__name__}")
        return None


def _transcript_via_ytdlp(video_id: str, lang: str) -> list[dict[str, Any]] | None:
    """Фолбэк: yt-dlp скачивает автосубтитры vtt, парсим их."""
    import tempfile

    import yt_dlp

    with tempfile.TemporaryDirectory() as tmp:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writeautomaticsub": True,
            "writesubtitles": True,
            "subtitleslangs": [lang, f"{lang}-orig"],
            "subtitlesformat": "vtt",
            "outtmpl": str(Path(tmp) / "%(id)s.%(ext)s"),
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        except Exception as exc:
            print(f"[info] yt-dlp субтитры не сработали для {video_id}: {type(exc).__name__}")
            return None
        vtts = sorted(Path(tmp).glob("*.vtt"))
        if not vtts:
            return None
        return parse_vtt(vtts[0].read_text(encoding="utf-8"))


_TS_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})")
_TAG_RE = re.compile(r"<[^>]+>")


def parse_vtt(vtt: str) -> list[dict[str, Any]]:
    """Парсер WebVTT в список {text, start, duration}. Дубликаты строк (roll-up автосубов) убираются."""
    snippets: list[dict[str, Any]] = []
    cur_start = cur_end = None
    cur_lines: list[str] = []

    def flush() -> None:
        nonlocal cur_start, cur_end, cur_lines
        if cur_start is not None and cur_lines:
            text = " ".join(cur_lines).strip()
            if text:
                snippets.append({"text": text, "start": cur_start, "duration": max(cur_end - cur_start, 0.0)})
        cur_start = cur_end = None
        cur_lines = []

    for line in vtt.splitlines():
        m = _TS_RE.search(line)
        if m:
            flush()
            h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(x) for x in m.groups())
            cur_start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
            cur_end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
            continue
        line = _TAG_RE.sub("", line).strip()
        if not line or line == "WEBVTT" or line.startswith(("Kind:", "Language:", "NOTE")):
            continue
        if cur_start is not None:
            cur_lines.append(line)
    flush()

    # roll-up автосубтитры дублируют строки между кью — дедупликация по совпадению текста
    deduped: list[dict[str, Any]] = []
    for s in snippets:
        if deduped and s["text"] == deduped[-1]["text"]:
            deduped[-1]["duration"] = s["start"] + s["duration"] - deduped[-1]["start"]
            continue
        if deduped and deduped[-1]["text"] and s["text"].startswith(deduped[-1]["text"]):
            tail = s["text"][len(deduped[-1]["text"]):].strip()
            if tail:
                s = {**s, "text": tail}
            else:
                continue
        deduped.append(dict(s))
    return deduped


_SENT_END = re.compile(r"[.!?…]\s*$")


def chunk_transcript(
    snippets: list[dict[str, Any]],
    target_chars: int = 600,
    max_chars: int = 750,
    pause_break: float = 2.0,
) -> list[dict[str, Any]]:
    """Склеивает реплики в связные фрагменты ~500-700 символов.

    Границу предпочитаем по концу предложения (.!?), при отсутствии пунктуации —
    по паузе > pause_break секунд между репликами. Повторы строк дедуплицируются.
    Возвращает [{text, start_sec, end_sec}].
    """
    chunks: list[dict[str, Any]] = []
    buf: list[str] = []
    buf_start: float | None = None
    buf_end: float = 0.0
    prev_text = ""

    def flush() -> None:
        nonlocal buf, buf_start, buf_end
        text = " ".join(buf).strip()
        text = re.sub(r"\s+", " ", text)
        if text and buf_start is not None:
            chunks.append({"text": text, "start_sec": int(buf_start), "end_sec": int(round(buf_end))})
        buf = []
        buf_start = None

    for i, s in enumerate(snippets):
        text = s["text"].strip()
        if not text or text == prev_text:
            continue
        prev_text = text
        start = float(s["start"])
        end = start + float(s.get("duration") or 0)

        if buf_start is None:
            buf_start = start
        # пауза между репликами
        if buf and start - buf_end > pause_break and sum(len(x) for x in buf) >= target_chars * 0.5:
            flush()
            buf_start = start
        buf.append(text)
        buf_end = end

        cur_len = sum(len(x) + 1 for x in buf)
        if cur_len >= target_chars and _SENT_END.search(text):
            flush()
        elif cur_len >= max_chars:
            flush()
    flush()
    return chunks
