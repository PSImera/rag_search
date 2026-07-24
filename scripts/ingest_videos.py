"""Пакетная индексация YouTube-видео (комментарии + транскрипции) в коллекцию.

Запуск:
    .\\.venv\\Scripts\\python.exe scripts\\ingest_videos.py docs\\<файл_со_ссылками> [--collection scientism]

Файл — список ссылок youtu.be / youtube.com/watch / youtube.com/live (по одной в строке,
пустые строки и строки на # игнорируются). Результат — сводная таблица и logs/ingest_report.json.

Пайплайн разбит на две фазы, чтобы не гонять GPU мелкими батчами:
  Фаза 1 (сеть): параллельно скачиваем инфо/комменты/субтитры всех видео в кэш data/youtube/.
  Фаза 2 (GPU):  собираем чанки со всех видео в один поток и эмбеддим полными батчами
                 EMBED_BATCH_SIZE — маленькие видео больше не простаивают GPU по одному.
Обе фазы идемпотентны: кэш переиспользуется, уже проиндексированные чанки пропускаются.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_MIN_CHARS, LOGS_ROOT
from src.embeddings import E5Embeddings
from src.index import (
    embed_pending,
    select_new,
    yt_comment_chunks,
    yt_live_chat_chunks,
    yt_transcript_chunks,
)
from src.vectorstore import VectorStore
from src.youtube import (
    chunk_transcript,
    expand_playlist,
    extract_video_id,
    fetch_comments,
    fetch_live_chat,
    fetch_transcript,
    fetch_video_info,
    is_playlist_url,
    is_stream,
    log_progress,
    set_cookie_source,
    set_throttle,
)


# пул позиций для tqdm-баров: фаза 1 качает несколько видео разом, каждому нужна своя строка.
# Размер = числу воркеров (см. main); слот берётся на время видео и возвращается после.
_slots: queue.Queue[int] = queue.Queue()


def download_one(
    video_id: str, lang: str, fetch_subs: bool = True, live_chat: bool = True
) -> dict:
    """Фаза 1 для одного видео: тянем всё в кэш, возвращаем краткий статус (без модели).

    fetch_subs=False (режим --asr) — субтитры не трогаем, транскрипт делается позже в фазе 1.5
    распознаванием аудио.
    live_chat=True (по умолчанию) — у СТРИМОВ (is_stream по live_status) дополнительно тянем
    реплей-чат в кэш; обычные ролики не трогаются (лишних запросов нет). --no-live-chat выключает.
    """
    info = fetch_video_info(video_id)
    title = (info.get("title") or "")[:30]
    stream = live_chat and is_stream(info)

    slot = _slots.get()
    bar: tqdm | None = None

    def on_progress(running: int, est: int) -> None:
        # бар создаём лениво — на первом реальном счётчике (кэш-хиты/видео без комментов бар не плодят)
        nonlocal bar
        total = max(est, running) or None
        if bar is None:
            bar = tqdm(
                total=total,
                position=slot,
                leave=False,
                desc=f"{video_id} «{title}»",
                unit=" комм.",
                bar_format="{desc}: {n_fmt}/{total_fmt} |{bar}| {percentage:3.0f}%",
            )
        if total != bar.total:
            bar.total = total
        bar.n = running
        bar.refresh()

    try:
        comments = fetch_comments(video_id, info, on_progress=on_progress)
    finally:
        if bar is not None:
            bar.close()
        _slots.put(slot)

    snippets = fetch_transcript(video_id, info, lang=lang) if fetch_subs else None
    chat = fetch_live_chat(video_id, info) if stream else None
    return {
        "video_id": video_id,
        "title": info["title"],
        "channel": info["channel"],
        "comments_raw": len(comments),
        "transcript_snippets": len(snippets or []),
        "is_stream": stream,
        "live_chat_raw": len(chat or []),
        "no_comments": not comments,
        "no_transcript": not snippets,
        "no_live_chat": stream and not chat,
    }


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "list_file",
        nargs="?",
        default="docs/video_for_seach_scientism.md",
        help="Файл со списком ссылок (по умолчанию docs/video_for_seach_scientism.md)",
    )
    ap.add_argument("--collection", default="scientism")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    ap.add_argument(
        "--download-workers",
        type=int,
        default=3,
        help="Параллельных скачиваний в фазе 1. С троттлингом (--sleep-requests-*) можно "
        "безопасно поднять до 3-4: каждое видео качается «вежливо», но несколько сразу.",
    )
    ap.add_argument(
        "--cookies-from-browser",
        default='chrome:Profile 2',
        help="Cookies для yt-dlp (обход бот-проверки): 'chrome', 'firefox', "
        "'chrome:Profile 2' и т.п. Браузер должен быть закрыт.",
    )
    ap.add_argument(
        "--cookies-file",
        default=None,
        help="Альтернатива: путь к cookies.txt (Netscape-формат)",
    )
    ap.add_argument(
        "--sleep-requests-min",
        type=float,
        default=0.5,
        help="Мин. пауза (сек) между сетевыми запросами yt-dlp, включая страницы комментов. "
        "Антибот-троттлинг: даёт безопасно поднять --download-workers. 0 — выключить.",
    )
    ap.add_argument(
        "--sleep-requests-max",
        type=float,
        default=2.0,
        help="Макс. пауза (сек) между запросами. Реальная пауза — случайная из [min,max], своя "
        "на каждое видео («хаотичность», чтобы паттерн не был ровным пулемётом).",
    )
    ap.add_argument(
        "--no-live-chat",
        action="store_true",
        help="Не индексировать реплей-чат стримов. По умолчанию чат тянется автоматически "
        "у видео, которые являются записями стримов (live_status), — обычные ролики не трогаются.",
    )
    ap.add_argument(
        "--asr",
        action="store_true",
        help="Игнорировать субтитры YouTube: транскрипт ВСЕХ видео строить распознаванием "
        "аудио через Whisper (GPU). Нужно, когда автосубтитры кривые.",
    )
    ap.add_argument(
        "--asr-model", default="large-v3", help="Модель faster-whisper для --asr"
    )
    args = ap.parse_args()

    set_cookie_source(args.cookies_from_browser, args.cookies_file)
    set_throttle(args.sleep_requests_min, args.sleep_requests_max)
    live_chat_enabled = not args.no_live_chat

    lines = [
        line.strip()
        for line in Path(args.list_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    urls: list[str] = []
    for line in lines:
        if is_playlist_url(line):
            expanded = expand_playlist(line)
            print(f"Плейлист {line} -> {len(expanded)} видео")
            urls.extend(expanded)
        else:
            urls.append(line)
    video_ids: list[str] = []
    for u in urls:
        try:
            video_ids.append(extract_video_id(u))
        except ValueError as e:
            print(f"[warn] {e}")

    print(f"Видео в списке: {len(video_ids)}, коллекция: {args.collection}")
    t_start = time.perf_counter()

    # ── Фаза 1: скачивание всех видео в кэш (сеть, параллельно) ──────────────────────
    print(
        f"\n{'='*80}\nФАЗА 1/2 — скачивание в кэш ({args.download_workers} потоков)\n{'='*80}"
    )
    status: dict[str, dict] = {}
    errors: dict[str, str] = {}
    # по слоту-строке на воркера: столько баров одновременно, сколько потоков качают
    for slot in range(args.download_workers):
        _slots.put(slot)
    with ThreadPoolExecutor(max_workers=args.download_workers) as ex:
        # в режиме --asr субтитры не качаем вообще — транскрипт делается в фазе 1.5 из аудио
        futs = {
            ex.submit(download_one, vid, args.lang, not args.asr, live_chat_enabled): vid
            for vid in video_ids
        }
        for done, fut in enumerate(as_completed(futs), 1):
            vid = futs[fut]
            try:
                st = status[vid] = fut.result()
                note = []
                if st["no_comments"]:
                    note.append("нет комментов")
                if st["no_transcript"]:
                    note.append("нет транскрипции")
                if st.get("no_live_chat"):
                    note.append("нет чата")
                tail = f"  ({', '.join(note)})" if note else ""
                chat_part = (
                    f" чат={st['live_chat_raw']}" if st.get("is_stream") else ""
                )
                log_progress(
                    f"[{done}/{len(video_ids)}] ✓ {vid}  «{st['title'][:50]}»  "
                    f"комм={st['comments_raw']} субтитры={st['transcript_snippets']}{chat_part}{tail}"
                )
            except Exception as exc:
                errors[vid] = str(exc)[:200]
                log_progress(f"[{done}/{len(video_ids)}] {vid}  [error] {exc}")

    # ── Фаза 1.5: ASR всех видео из аудио (--asr игнорирует субтитры YouTube) ──────────
    # Последовательно, модель Whisper грузится один раз; готовый непустой кэш пропускается.
    if args.asr:
        asr_ids = [vid for vid in video_ids if vid not in errors]
        if asr_ids:
            print(
                f"\n{'='*80}\nФАЗА 1.5 — ASR из аудио ({len(asr_ids)} видео, субтитры YouTube игнорируются)\n{'='*80}"
            )
            for done, vid in enumerate(asr_ids, 1):
                info = fetch_video_info(vid)
                print(f"[{done}/{len(asr_ids)}] ASR: {vid} «{info['title'][:50]}»")
                snippets = fetch_transcript(
                    vid,
                    info,
                    lang=args.lang,
                    force_asr=True,
                    asr_model=args.asr_model,
                )
                status[vid]["transcript_snippets"] = len(snippets or [])
                status[vid]["no_transcript"] = not snippets

    # ── Фаза 2: сборка чанков со всех видео и эмбеддинг полными батчами (GPU) ─────────
    print(f"\n{'='*80}\nФАЗА 2/2 — эмбеддинг (полные батчи по всем видео)\n{'='*80}")
    print("Загрузка модели эмбеддингов...")
    emb = E5Embeddings()
    store = VectorStore(args.collection, dim=emb.dim)
    store.ensure_index(emb.dim)
    print(
        f"Модель загружена: {emb.model.device} | уже в индексе векторов: {store.index.ntotal if store.index else 0}"
    )

    seen = store.existing_ids()
    pending: list = []
    report: list[dict] = []
    for vid in video_ids:
        if vid in errors:
            report.append({"video_id": vid, "error": errors[vid]})
            continue
        st = status[vid]
        info = fetch_video_info(vid)  # кэш-хит, без сети
        entry = {
            "video_id": vid,
            "title": st["title"],
            "channel": st["channel"],
            "comments_raw": st["comments_raw"],
            "comment_chunks": 0,
            "transcript_chunks": 0,
            "live_chat_chunks": 0,
        }

        # кэш-хит для уже готовых (фаза 1 или 1.5); force_asr держит семантику --asr,
        # если у видео ещё нет кэша (ASR в 1.5 сорвался) — субтитры YouTube не подхватятся
        snippets = fetch_transcript(
            vid, info, lang=args.lang, force_asr=args.asr, asr_model=args.asr_model
        )
        if snippets:
            raw_rows, chunks = yt_transcript_chunks(
                vid, info, chunk_transcript(snippets), args.lang
            )
            store.raw.upsert_yt_chunks(raw_rows)
            new = select_new(chunks, seen)
            entry["transcript_chunks"] = len(new)
            pending.extend(new)

        comments = fetch_comments(vid, info)
        if comments:
            raw_rows, chunks = yt_comment_chunks(vid, info, comments, args.min_chars)
            store.raw.upsert_yt_comments(raw_rows)
            new = select_new(chunks, seen)
            entry["comment_chunks"] = len(new)
            pending.extend(new)

        if live_chat_enabled and is_stream(info):
            messages = fetch_live_chat(vid, info)  # кэш-хит из фазы 1, без сети
            if messages:
                raw_rows, chunks = yt_live_chat_chunks(
                    vid, info, messages, args.min_chars
                )
                store.raw.upsert_yt_live_chat(raw_rows)
                new = select_new(chunks, seen)
                entry["live_chat_chunks"] = len(new)
                pending.extend(new)

        report.append(entry)

    print(f"Новых чанков к эмбеддингу: {len(pending)} (со всех видео)")
    save_meta = {"source": args.list_file, "collection": args.collection}
    embedded = embed_pending(
        store, emb, pending, "Эмбеддинг (все видео)", save_meta=save_meta
    )
    store.save(save_meta)
    total_vectors = store.index.ntotal if store.index else 0
    store.close()

    # ── Отчёт ────────────────────────────────────────────────────────────────────────
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    out = LOGS_ROOT / "ingest_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t_start
    print(f"\n\n{'='*100}\nСВОДКА\n{'='*100}")
    chat_hdr = f" {'чат.ч':>7}" if live_chat_enabled else ""
    print(
        f"{'video_id':<13} {'канал':<24} {'название':<38} {'комм.ч':>7} {'транскр.ч':>9}{chat_hdr}  примечание"
    )
    for e in report:
        note = e.get("error", "")
        title = (e.get("title") or "")[:36]
        channel = (e.get("channel") or "")[:22]
        chat_col = f" {e.get('live_chat_chunks', 0):>7}" if live_chat_enabled else ""
        print(
            f"{e['video_id']:<13} {channel:<24} {title:<38} "
            f"{e.get('comment_chunks', 0):>7} {e.get('transcript_chunks', 0):>9}{chat_col}  {note}"
        )
    print(
        f"\nНовых чанков в индексе: {embedded} | всего векторов: {total_vectors} | "
        f"ошибок скачивания: {len(errors)}"
    )
    print(f"Время: {elapsed/60:.1f} мин ({elapsed:.0f} сек)")
    print(f"Отчёт: {out}")


if __name__ == "__main__":
    main()
