from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tqdm import tqdm

from src.config import DEFAULT_COLLECTION, DEFAULT_LINK_TEMPLATE, EMBED_BATCH_SIZE
from src.embeddings import E5Embeddings
from src.parser import iter_messages, make_link, read_chat_header
from src.vectorstore import VectorStore, chunk_id


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
    pending: list[dict] = []  # сообщения на эмбеддинг (len>=min_chars и ещё не в индексе)
    total_msgs = 0
    embedded = 0
    skipped_existing = 0

    def flush_embed(batch: list[dict]) -> None:
        nonlocal embedded
        if not batch:
            return
        vectors = emb.embed_passages([m["text"] for m in batch])
        ids = [m["_id"] for m in batch]
        metas = [
            {
                "source": "tg",
                "text": m["text"],
                "author": m["author"],
                "author_id": m["author_id"],
                "date": m["date"],
                "message_id": m["message_id"],
                "chat_id": chat_id_str,
                "chat_name": chat_name,
                "reply_to_message_id": m["reply_to_message_id"],
                "link": m["link"],
            }
            for m in batch
        ]
        store.add_batch(ids, vectors, metas)
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
        cid = chunk_id(chat_id, msg["message_id"])
        if cid in already:
            skipped_existing += 1
            continue
        msg["_id"] = cid
        msg["link"] = link
        pending.append(msg)
        already.add(cid)  # защита от дублей внутри прохода
        if len(pending) >= batch_size:
            flush_embed(pending)
            pending = []
            progress.set_postfix_str(f"embedded={embedded}")
            if embedded % (batch_size * 20) == 0:
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


def add_yt_comments(url_or_id: str, collection: str, min_chars: int = 10) -> dict:
    """Индексация комментариев YouTube-видео. Чанк = один комментарий."""
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
    already = store.existing_ids()

    raw_rows = []
    pending = []
    embedded = 0
    for c in comments:
        link = f"https://www.youtube.com/watch?v={video_id}&lc={c['comment_id']}"
        raw_rows.append({**c, "video_id": video_id, "link": link})
        if len(c["text"]) < min_chars:
            continue
        cid = chunk_id(f"yt:{video_id}", c["comment_id"])
        if cid in already:
            continue
        already.add(cid)
        pending.append((cid, c, link))
    store.raw.upsert_yt_comments(raw_rows)

    for i in tqdm(range(0, len(pending), EMBED_BATCH_SIZE), desc="Эмбеддинг комментариев", unit="batch"):
        batch = pending[i : i + EMBED_BATCH_SIZE]
        vectors = emb.embed_passages([c["text"] for _, c, _ in batch])
        ids = [cid for cid, _, _ in batch]
        metas = [
            {
                "source": "yt_comments",
                "text": c["text"],
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
            for _, c, link in batch
        ]
        store.add_batch(ids, vectors, metas)
        embedded += len(batch)

    store.save()
    store.close()
    print(f"Новых комментариев в индексе: {embedded} (всего получено {len(comments)})")
    return {"video_id": video_id, "title": info["title"], "channel": info["channel"], "comments_indexed": embedded, "comments_total": len(comments)}


def add_yt_transcript(url_or_id: str, collection: str, lang: str = "ru") -> dict:
    """Индексация транскрипции YouTube-видео. Чанки ~500-700 символов с таймингами."""
    from src.youtube import chunk_transcript, extract_video_id, fetch_transcript, fetch_video_info

    video_id = extract_video_id(url_or_id)
    info = fetch_video_info(video_id)
    print(f"Видео: {info['channel']} — {info['title']} [{video_id}]")
    snippets = fetch_transcript(video_id, info, lang=lang)
    if not snippets:
        print("[skip] Транскрипция недоступна (стрим без субтитров или субтитры отключены)")
        return {"video_id": video_id, "title": info["title"], "channel": info["channel"], "transcript_chunks": 0, "skipped": "no_transcript"}

    chunks = chunk_transcript(snippets)
    print(f"Реплик: {len(snippets)} -> чанков: {len(chunks)}")

    emb = E5Embeddings()
    store = VectorStore(collection, dim=emb.dim)
    store.ensure_index(emb.dim)
    already = store.existing_ids()

    raw_rows = []
    pending = []
    embedded = 0
    for ch in chunks:
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
        if cid in already:
            continue
        already.add(cid)
        pending.append((cid, ch, link))
    store.raw.upsert_yt_chunks(raw_rows)

    for i in tqdm(range(0, len(pending), EMBED_BATCH_SIZE), desc="Эмбеддинг транскрипции", unit="batch"):
        batch = pending[i : i + EMBED_BATCH_SIZE]
        vectors = emb.embed_passages([ch["text"] for _, ch, _ in batch])
        ids = [cid for cid, _, _ in batch]
        metas = [
            {
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
            for _, ch, link in batch
        ]
        store.add_batch(ids, vectors, metas)
        embedded += len(batch)

    store.save()
    store.close()
    print(f"Новых чанков транскрипции в индексе: {embedded} (всего чанков {len(chunks)})")
    return {"video_id": video_id, "title": info["title"], "channel": info["channel"], "transcript_chunks": embedded, "chunks_total": len(chunks)}


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

    add = sub.add_parser("add", help="Добавить Telegram-чат в индекс (идемпотентно)")
    add.add_argument("json_path", help="Путь к JSON-экспорту Telegram Desktop")
    add.add_argument("--collection", default=DEFAULT_COLLECTION)
    add.add_argument("--min-chars", type=int, default=80, help="Минимум символов для эмбеддинга")
    add.add_argument("--link-template", default=DEFAULT_LINK_TEMPLATE)
    add.add_argument("--username", default=None, help="Публичный username чата (даёт t.me/<username>/<id>)")
    add.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE)

    yc = sub.add_parser("add-yt-comments", help="Добавить комментарии YouTube-видео")
    yc.add_argument("url", help="URL или video_id")
    yc.add_argument("--collection", default=DEFAULT_COLLECTION)
    yc.add_argument("--min-chars", type=int, default=10)

    yt = sub.add_parser("add-yt-transcript", help="Добавить транскрипцию YouTube-видео")
    yt.add_argument("url", help="URL или video_id")
    yt.add_argument("--collection", default=DEFAULT_COLLECTION)
    yt.add_argument("--lang", default="ru")

    sub.add_parser("list", help="Список коллекций со статистикой")
    return p


def main() -> None:
    _reconfigure_stdout()
    args = build_parser().parse_args()
    if args.command == "add":
        add_chat(
            json_path=Path(args.json_path),
            collection=args.collection,
            min_chars=args.min_chars,
            link_template=args.link_template,
            username=args.username,
            batch_size=args.batch_size,
        )
    elif args.command == "add-yt-comments":
        add_yt_comments(args.url, args.collection, args.min_chars)
    elif args.command == "add-yt-transcript":
        add_yt_transcript(args.url, args.collection, args.lang)
    elif args.command == "list":
        list_collections()


if __name__ == "__main__":
    main()
