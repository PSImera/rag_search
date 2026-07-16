"""Пакетная индексация YouTube-видео (комментарии + транскрипции) в коллекцию.

Запуск:
    .\\.venv\\Scripts\\python.exe scripts\\ingest_videos.py docs\\<файл_со_ссылками> [--collection scientism]

Файл — список ссылок youtu.be / youtube.com/watch / youtube.com/live (по одной в строке,
пустые строки игнорируются). Результат — сводная таблица и logs/ingest_report.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LOGS_ROOT
from src.index import add_yt_comments, add_yt_transcript
from src.youtube import extract_video_id


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("list_file")
    ap.add_argument("--collection", default="scientism")
    ap.add_argument("--lang", default="ru")
    args = ap.parse_args()

    urls = [
        line.strip()
        for line in Path(args.list_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    video_ids = []
    for u in urls:
        try:
            video_ids.append(extract_video_id(u))
        except ValueError as e:
            print(f"[warn] {e}")

    print(f"Видео в списке: {len(video_ids)}, коллекция: {args.collection}")
    report = []
    for n, vid in enumerate(video_ids, 1):
        print(f"\n{'='*80}\n[{n}/{len(video_ids)}] {vid}\n{'='*80}")
        entry: dict = {"video_id": vid}
        t0 = time.perf_counter()
        try:
            r = add_yt_transcript(vid, args.collection, lang=args.lang)
            entry.update({"title": r.get("title"), "channel": r.get("channel"),
                          "transcript_chunks": r.get("transcript_chunks", 0),
                          "transcript_skipped": r.get("skipped")})
        except Exception as exc:
            print(f"[error] транскрипция {vid}: {exc}")
            entry["transcript_error"] = str(exc)[:200]
        try:
            r = add_yt_comments(vid, args.collection)
            entry.update({"title": entry.get("title") or r.get("title"),
                          "channel": entry.get("channel") or r.get("channel"),
                          "comments_indexed": r.get("comments_indexed", 0),
                          "comments_total": r.get("comments_total", 0),
                          "comments_skipped": r.get("skipped")})
        except Exception as exc:
            print(f"[error] комментарии {vid}: {exc}")
            entry["comments_error"] = str(exc)[:200]
        entry["seconds"] = round(time.perf_counter() - t0, 1)
        report.append(entry)

    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    out = LOGS_ROOT / "ingest_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n\n{'='*100}\nСВОДКА\n{'='*100}")
    print(f"{'video_id':<13} {'канал':<25} {'название':<40} {'комм.':>6} {'транскр.':>8}  примечание")
    for e in report:
        note = e.get("transcript_skipped") or e.get("transcript_error") or ""
        cnote = e.get("comments_skipped") or e.get("comments_error") or ""
        if cnote:
            note = (note + "; " if note else "") + f"comments: {cnote}"
        title = (e.get("title") or "")[:38]
        channel = (e.get("channel") or "")[:23]
        print(f"{e['video_id']:<13} {channel:<25} {title:<40} {e.get('comments_indexed',0):>6} {e.get('transcript_chunks',0):>8}  {note}")
    print(f"\nОтчёт: {out}")


if __name__ == "__main__":
    main()
