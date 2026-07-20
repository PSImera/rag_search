from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tqdm import tqdm

from src.config import (
    DEFAULT_COLLECTION,
    DEFAULT_LINK_TEMPLATE,
    DEFAULT_MIN_CHARS,
    EMBED_BATCH_SIZE,
    MAX_CHUNK_CHARS,
)
from src.embeddings import E5Embeddings
from src.parser import iter_messages, make_link, read_chat_header
from src.vectorstore import VectorStore, chunk_id

# Тип чанка на эмбеддинг: (int64-id, текст для энкодера, метаданные для docstore).
Chunk = tuple[int, str, dict]


def _reconfigure_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def add_chat(
    json_path: Path,
    collection: str,
    min_chars: int,
    link_template: str,
    username: str | None = None,
    batch_size: int = EMBED_BATCH_SIZE,
) -> None:
    started = time.perf_counter()
    header = read_chat_header(json_path)
    chat_id = header.get("id")
    chat_name = header.get("name") or ""
    chat_id_str = str(chat_id)
    print(f"Чат: {chat_name} (id={chat_id}, type={header.get('type')})")
    print(f"Коллекция: {collection} | min-chars={min_chars} | link-template={link_template}")

    from src.youtube import split_long_text

    print("Загрузка модели эмбеддингов...")
    emb = E5Embeddings()
    print(f"Модель загружена, устройство: {emb.model.device}, размерность: {emb.dim}")

    store = VectorStore(collection, dim=emb.dim)
    store.ensure_index(emb.dim)
    already = store.existing_ids()
    print(f"Уже в индексе (векторов): {len(already)}")

    # Первый проход: все сообщения -> SQLite (для контекста), сбор кандидатов на эмбеддинг.
    print("Проход по сообщениям...")
    raw_buffer: list[dict] = []
    pending: list[Chunk] = []  # чанки на эмбеддинг (len>=min_chars и ещё не в индексе)
    total_msgs = 0
    embedded = 0
    skipped_existing = 0
    flush_count = 0

    def flush_embed(batch: list[Chunk]) -> None:
        nonlocal embedded
        if not batch:
            return
        vectors = emb.embed_passages([text for _, text, _ in batch])
        store.add_batch([cid for cid, _, _ in batch], vectors, [meta for _, _, meta in batch])
        embedded += len(batch)

    progress = tqdm(iter_messages(json_path), desc="Сообщения", unit="msg")
    for msg in progress:
        total_msgs += 1
        link = make_link(link_template, chat_id, username, msg["message_id"])
        raw_buffer.append(
            {
                "chat_id": chat_id_str,
                "message_id": msg["message_id"],
                "author": msg["author"],
                "author_id": msg["author_id"],
                "date": msg["date"],
                "reply_to": msg["reply_to_message_id"],
                "text": msg["text"],
                "link": link,
            }
        )
        if len(raw_buffer) >= 5000:
            store.raw.upsert_many(raw_buffer)
            raw_buffer.clear()

        if len(msg["text"]) < min_chars:
            continue
        # Длинные сообщения (графомания) дробим на чанки: одиночные сохраняют прежний id
        # (yt-аналогично: chat_id:message_id) — идемпотентность старой индексации цела.
        pieces = split_long_text(msg["text"], MAX_CHUNK_CHARS)
        for idx, piece in enumerate(pieces):
            key = msg["message_id"] if len(pieces) == 1 else f"{msg['message_id']}#{idx}"
            cid = chunk_id(chat_id, key)
            if cid in already:
                skipped_existing += 1
                continue
            already.add(cid)  # защита от дублей внутри прохода
            pending.append((cid, piece, {
                "source": "tg",
                "text": piece,
                "author": msg["author"],
                "author_id": msg["author_id"],
                "date": msg["date"],
                "message_id": msg["message_id"],
                "chat_id": chat_id_str,
                "chat_name": chat_name,
                "reply_to_message_id": msg["reply_to_message_id"],
                "link": link,
            }))
        if len(pending) >= batch_size:
            flush_embed(pending)
            pending = []
            flush_count += 1
            progress.set_postfix_str(f"embedded={embedded}")
            if flush_count % 20 == 0:
                store.save({"source": str(json_path), "chat_name": chat_name})

    if raw_buffer:
        store.raw.upsert_many(raw_buffer)
    flush_embed(pending)
    store.save({"source": str(json_path), "chat_name": chat_name})
    raw_count = store.raw.count()
    total_vectors = store.index.ntotal if store.index else 0
    store.close()

    elapsed = time.perf_counter() - started
    print("-" * 60)
    print(f"Всего непустых сообщений: {total_msgs}")
    print(f"Сырых сообщений в SQLite: {raw_count}")
    print(f"Новых эмбеддингов добавлено: {embedded}")
    print(f"Пропущено (уже в индексе): {skipped_existing}")
    print(f"Всего векторов в индексе: {total_vectors}")
    print(f"Время: {elapsed/60:.1f} мин ({elapsed:.0f} сек)")


def yt_comment_chunks(video_id: str, info: dict, comments: list[dict], min_chars: int) -> tuple[list[dict], list[Chunk]]:
    """Комментарии -> (raw_rows для SQLite, чанки на эмбеддинг). Без сети и без модели.

    Длинные комментарии дробятся на несколько чанков (см. MAX_CHUNK_CHARS). Комментарий,
    уложившийся в один чанк, сохраняет прежний id (yt:<vid>:<comment_id>) — идемпотентность
    старой индексации не ломается; дроблёные получают суффикс #0, #1, ...
    """
    from src.youtube import split_long_text

    raw_rows: list[dict] = []
    chunks: list[Chunk] = []
    for c in comments:
        link = f"https://www.youtube.com/watch?v={video_id}&lc={c['comment_id']}"
        raw_rows.append({**c, "video_id": video_id, "link": link})
        if len(c["text"]) < min_chars:
            continue
        pieces = split_long_text(c["text"], MAX_CHUNK_CHARS)
        for idx, piece in enumerate(pieces):
            key = c["comment_id"] if len(pieces) == 1 else f"{c['comment_id']}#{idx}"
            cid = chunk_id(f"yt:{video_id}", key)
            meta = {
                "source": "yt_comments",
                "text": piece,
                "author": c["author"],
                "author_id": c["author_id"],
                "date": c["date"],
                "likes": c["likes"],
                "video_id": video_id,
                "video_title": info["title"],
                "channel": info["channel"],
                "comment_id": c["comment_id"],
                "parent_id": c["parent_id"],
                "link": link,
            }
            chunks.append((cid, piece, meta))
    return raw_rows, chunks


def select_new(chunks: list[Chunk], seen: set[int]) -> list[Chunk]:
    """Отбрасывает чанки, чей id уже в seen (уже в индексе или уже собран); seen дополняется."""
    out: list[Chunk] = []
    for chunk in chunks:
        if chunk[0] in seen:
            continue
        seen.add(chunk[0])
        out.append(chunk)
    return out


def embed_pending(
    store: VectorStore,
    emb: E5Embeddings,
    pending: list[Chunk],
    desc: str,
    *,
    save_meta: dict | None = None,
    save_every: int = 10,
) -> int:
    """Эмбеддит собранные чанки полными батчами EMBED_BATCH_SIZE и пишет их в store.

    Периодический store.save (раз в save_every батчей) делает индексацию устойчивой к обрыву:
    при повторном запуске уже сохранённые чанки отфильтруются через existing_ids.
    """
    embedded = 0
    for bi, i in enumerate(tqdm(range(0, len(pending), EMBED_BATCH_SIZE), desc=desc, unit="batch")):
        batch = pending[i : i + EMBED_BATCH_SIZE]
        vectors = emb.embed_passages([text for _, text, _ in batch])
        store.add_batch([cid for cid, _, _ in batch], vectors, [meta for _, _, meta in batch])
        embedded += len(batch)
        if (bi + 1) % save_every == 0:
            store.save(save_meta)
    return embedded


def add_yt_comments(url_or_id: str, collection: str, min_chars: int = DEFAULT_MIN_CHARS) -> dict:
    """Индексация комментариев YouTube-видео. Чанк = комментарий (длинные дробятся)."""
    from src.youtube import extract_video_id, fetch_comments, fetch_video_info

    video_id = extract_video_id(url_or_id)
    info = fetch_video_info(video_id)
    print(f"Видео: {info['channel']} — {info['title']} [{video_id}]")
    comments = fetch_comments(video_id, info)
    print(f"Комментариев получено: {len(comments)}")
    if not comments:
        print("[skip] Комментарии отсутствуют или отключены")
        return {"video_id": video_id, "title": info["title"], "channel": info["channel"], "comments_indexed": 0, "skipped": "no_comments"}

    emb = E5Embeddings()
    store = VectorStore(collection, dim=emb.dim)
    store.ensure_index(emb.dim)

    raw_rows, chunks = yt_comment_chunks(video_id, info, comments, min_chars)
    store.raw.upsert_yt_comments(raw_rows)
    pending = select_new(chunks, store.existing_ids())
    embedded = embed_pending(store, emb, pending, "Эмбеддинг комментариев")

    store.save()
    store.close()
    print(f"Новых чанков комментариев в индексе: {embedded} (всего получено {len(comments)})")
    return {"video_id": video_id, "title": info["title"], "channel": info["channel"], "comments_indexed": embedded, "comments_total": len(comments)}


def yt_transcript_chunks(video_id: str, info: dict, chunks_in: list[dict], lang: str) -> tuple[list[dict], list[Chunk]]:
    """Чанки транскрипции (из chunk_transcript) -> (raw_rows, чанки на эмбеддинг). Без сети/модели."""
    raw_rows: list[dict] = []
    chunks: list[Chunk] = []
    for ch in chunks_in:
        link = f"https://youtu.be/{video_id}?t={ch['start_sec']}"
        raw_rows.append(
            {
                "video_id": video_id,
                "start_sec": ch["start_sec"],
                "end_sec": ch["end_sec"],
                "text": ch["text"],
                "link": link,
                "video_title": info["title"],
                "channel": info["channel"],
                "lang": lang,
            }
        )
        cid = chunk_id(f"yt:{video_id}", f"t{ch['start_sec']}")
        meta = {
            "source": "yt_transcript",
            "text": ch["text"],
            "video_id": video_id,
            "video_title": info["title"],
            "channel": info["channel"],
            "lang": lang,
            "start_sec": ch["start_sec"],
            "end_sec": ch["end_sec"],
            "link": link,
        }
        chunks.append((cid, ch["text"], meta))
    return raw_rows, chunks


def add_yt_transcript(
    url_or_id: str, collection: str, lang: str = "ru", asr: bool = False, asr_model: str = "large-v3"
) -> dict:
    """Индексация транскрипции YouTube-видео. Чанки ~500-700 символов с таймингами.

    asr=True — если у видео нет ни ручных, ни автоматических субтитров, распознаёт
    аудио локально через Whisper (см. src.youtube.transcribe_with_whisper).
    """
    from src.youtube import chunk_transcript, extract_video_id, fetch_transcript, fetch_video_info

    video_id = extract_video_id(url_or_id)
    info = fetch_video_info(video_id)
    print(f"Видео: {info['channel']} — {info['title']} [{video_id}]")
    snippets = fetch_transcript(video_id, info, lang=lang, use_asr=asr, asr_model=asr_model)
    if not snippets:
        print("[skip] Транскрипция недоступна (стрим без субтитров или субтитры отключены)")
        return {"video_id": video_id, "title": info["title"], "channel": info["channel"], "transcript_chunks": 0, "skipped": "no_transcript"}

    chunks_in = chunk_transcript(snippets)
    print(f"Реплик: {len(snippets)} -> чанков: {len(chunks_in)}")

    emb = E5Embeddings()
    store = VectorStore(collection, dim=emb.dim)
    store.ensure_index(emb.dim)

    raw_rows, chunks = yt_transcript_chunks(video_id, info, chunks_in, lang)
    store.raw.upsert_yt_chunks(raw_rows)
    pending = select_new(chunks, store.existing_ids())
    embedded = embed_pending(store, emb, pending, "Эмбеддинг транскрипции")

    store.save()
    store.close()
    print(f"Новых чанков транскрипции в индексе: {embedded} (всего чанков {len(chunks_in)})")
    return {"video_id": video_id, "title": info["title"], "channel": info["channel"], "transcript_chunks": embedded, "chunks_total": len(chunks_in)}


def delete_source(
    collection: str,
    *,
    video_id: str | None = None,
    chat_id: str | None = None,
    assume_yes: bool = False,
) -> dict:
    """Удаляет источник (видео или ТГ-чат) из коллекции: FAISS-индекс + docstore + SQLite.

    Пересоздание коллекции не требуется, GPU не задействуется (эмбеддинги не считаются).
    Типичный сценарий чистки «сирот» после смены схемы чанков: delete-source нужного видео,
    затем повторная индексация его же — она заново соберёт чанки по актуальной схеме.
    """
    from collections import Counter

    if bool(video_id) == bool(chat_id):
        raise ValueError("Укажи ровно одно: video_id ИЛИ chat_id")

    store = VectorStore(collection)
    if store.index is None:
        print(f"Коллекция «{collection}» пуста или не существует")
        store.close()
        return {"collection": collection, "removed_vectors": 0}

    if video_id:
        label = f"видео {video_id}"
        ids = [i for i, m in store.docstore.items() if str(m.get("video_id")) == str(video_id)]
    else:
        label = f"ТГ-чат {chat_id}"
        ids = [i for i, m in store.docstore.items()
               if m.get("source") == "tg" and str(m.get("chat_id")) == str(chat_id)]

    by_src = dict(Counter(store.docstore[i].get("source") for i in ids))
    print(f"К удалению из «{collection}» ({label}): {len(ids)} векторов {by_src}")

    if not assume_yes:
        resp = input("Удалить безвозвратно? [y/N] ").strip().lower()
        if resp != "y":
            print("Отменено, ничего не тронуто")
            store.close()
            return {"collection": collection, "removed_vectors": 0, "cancelled": True}

    removed = store.remove_ids(ids)
    if video_id:
        raw_c, raw_ch = store.raw.delete_video(str(video_id))
        print(f"SQLite: удалено комментариев {raw_c}, чанков транскрипции {raw_ch}")
    else:
        raw_msgs = store.raw.delete_chat(str(chat_id))
        print(f"SQLite: удалено сообщений {raw_msgs}")

    store.save()
    total = store.index.ntotal if store.index else 0
    store.close()
    print(f"Удалено векторов: {removed}. Осталось в индексе: {total}")
    return {"collection": collection, "removed_vectors": removed, "by_source": by_src, "remaining": total}


def list_collections() -> None:
    """Список коллекций со статистикой по источникам."""
    from src.config import STORE_ROOT

    if not STORE_ROOT.exists():
        print("Коллекций нет (папка store/ отсутствует)")
        return
    for coll_dir in sorted(STORE_ROOT.iterdir()):
        if not coll_dir.is_dir():
            continue
        name = coll_dir.name
        store = VectorStore(name)
        vectors = store.index.ntotal if store.index else 0
        by_source: dict[str, int] = {}
        chat_names: dict[str, str] = {}
        for meta in store.docstore.values():
            src = meta.get("source", "tg")
            by_source[src] = by_source.get(src, 0) + 1
            if src == "tg" and meta.get("chat_id"):
                chat_names[str(meta["chat_id"])] = meta.get("chat_name", "")
        raw_stats = store.raw.stats()
        print(f"\n=== Коллекция: {name} ===")
        print(f"Векторов в индексе: {vectors}")
        for src, n in sorted(by_source.items()):
            print(f"  {src}: {n}")
        for chat in raw_stats["tg_chats"]:
            label = chat_names.get(str(chat["chat_id"]), "")
            print(f"  [tg] {label} (id={chat['chat_id']}): {chat['messages']} сырых сообщений")
        for v in raw_stats["yt_videos_comments"]:
            print(f"  [yt_comments] {v['video_id']}: {v['comments']} комментариев (raw)")
        for v in raw_stats["yt_videos_chunks"]:
            print(f"  [yt_transcript] {v['video_id']} «{v['title']}»: {v['chunks']} чанков (raw)")
        store.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Индексация Telegram-чатов и YouTube в RAG")
    sub = p.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add-tg-chat", help="Добавить Telegram-чат в индекс (идемпотентно)")
    add.add_argument("json_path", help="Путь к JSON-экспорту Telegram Desktop")
    add.add_argument("--collection", default=DEFAULT_COLLECTION)
    add.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS, help="Минимум символов для эмбеддинга")
    add.add_argument("--link-template", default=DEFAULT_LINK_TEMPLATE)
    add.add_argument("--username", default=None, help="Публичный username чата (даёт t.me/<username>/<id>)")
    add.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE)

    yc = sub.add_parser("add-yt-comments", help="Добавить комментарии YouTube-видео")
    yc.add_argument("url", help="URL или video_id")
    yc.add_argument("--collection", default=DEFAULT_COLLECTION)
    yc.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    yc.add_argument("--cookies-from-browser", default=None,
                    help="Cookies для yt-dlp (обход бот-проверки): 'chrome', 'chrome:Profile 2', 'firefox'. Браузер закрыт.")
    yc.add_argument("--cookies-file", default=None, help="Альтернатива: путь к cookies.txt")

    yt = sub.add_parser("add-yt-transcript", help="Добавить транскрипцию YouTube-видео")
    yt.add_argument("url", help="URL или video_id")
    yt.add_argument("--collection", default=DEFAULT_COLLECTION)
    yt.add_argument("--lang", default="ru")
    yt.add_argument("--asr", action="store_true",
                    help="Если субтитров нет (ни ручных, ни авто) — распознать аудио локально через Whisper (GPU)")
    yt.add_argument("--asr-model", default="large-v3", help="Модель faster-whisper для --asr")
    yt.add_argument("--cookies-from-browser", default=None,
                    help="Cookies для yt-dlp (обход бот-проверки): 'chrome', 'chrome:Profile 2', 'firefox'. Браузер закрыт.")
    yt.add_argument("--cookies-file", default=None, help="Альтернатива: путь к cookies.txt")

    dl = sub.add_parser("delete-source", help="Удалить видео или ТГ-чат из коллекции (без пересоздания, без GPU)")
    dl.add_argument("--collection", default=DEFAULT_COLLECTION)
    grp = dl.add_mutually_exclusive_group(required=True)
    grp.add_argument("--video", help="video_id — удалить комментарии + транскрипцию видео")
    grp.add_argument("--chat", help="chat_id — удалить все сообщения ТГ-чата")
    dl.add_argument("--yes", action="store_true", help="Не спрашивать подтверждение")

    sub.add_parser("list", help="Список коллекций со статистикой")
    return p


def main() -> None:
    _reconfigure_stdout()
    args = build_parser().parse_args()
    if args.command == "add-tg-chat":
        add_chat(
            json_path=Path(args.json_path),
            collection=args.collection,
            min_chars=args.min_chars,
            link_template=args.link_template,
            username=args.username,
            batch_size=args.batch_size,
        )
    elif args.command == "add-yt-comments":
        from src.youtube import set_cookie_source
        set_cookie_source(args.cookies_from_browser, args.cookies_file)
        add_yt_comments(args.url, args.collection, args.min_chars)
    elif args.command == "add-yt-transcript":
        from src.youtube import set_cookie_source
        set_cookie_source(args.cookies_from_browser, args.cookies_file)
        add_yt_transcript(args.url, args.collection, args.lang, asr=args.asr, asr_model=args.asr_model)
    elif args.command == "delete-source":
        delete_source(args.collection, video_id=args.video, chat_id=args.chat, assume_yes=args.yes)
    elif args.command == "list":
        list_collections()


if __name__ == "__main__":
    main()
