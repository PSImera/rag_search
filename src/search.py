from __future__ import annotations

import argparse
import json
import sys

from src.config import DEFAULT_COLLECTION
from src.embeddings import E5Embeddings
from src.vectorstore import VectorStore


def _reconfigure_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def _mmss(sec: int) -> str:
    sec = int(sec)
    if sec >= 3600:
        return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _fmt_msg(m: dict, prefix: str = "    ") -> str:
    return f"{prefix}[{m.get('date','')}] {m.get('author','')} (msg {m.get('message_id')}): {m.get('text','')}"


def _get_context(store: VectorStore, r: dict, m: int) -> dict:
    src = r.get("source", "tg")
    if src == "yt_comments":
        return store.raw.get_comment_thread(r["video_id"], r["comment_id"], m)
    if src == "yt_transcript":
        return store.raw.get_chunk_context(r["video_id"], int(r["start_sec"]), m)
    return store.raw.get_context(str(r["chat_id"]), int(r["message_id"]), m)


def _print_result(store: VectorStore, i: int, r: dict, context: int) -> None:
    src = r.get("source", "tg")
    print(f"\n#{i}  score={r['score']:.4f}  [{src}]")
    if src == "yt_transcript":
        print(f"Видео: {r.get('channel','')} — {r.get('video_title','')}")
        print(f"Фрагмент: [{_mmss(r['start_sec'])}–{_mmss(r['end_sec'])}]")
        print(f"Текст: {r.get('text','')}")
        print(f"Ссылка: {r.get('link','')}")
        if context > 0:
            ctx = _get_context(store, r, context)
            for label, items in (("до", ctx["before"]), ("после", ctx["after"])):
                if items:
                    print(f"  --- {label} ---")
                    for c in items:
                        print(f"    [{_mmss(c['start_sec'])}–{_mmss(c['end_sec'])}] {c['text']}")
        return
    if src == "yt_comments":
        print(f"Видео: {r.get('channel','')} — {r.get('video_title','')}")
        print(f"Автор: {r.get('author','')}   Дата: {r.get('date','')}   Лайков: {r.get('likes',0)}")
        print(f"Текст: {r.get('text','')}")
        print(f"Ссылка: {r.get('link','')}")
        if context > 0:
            ctx = _get_context(store, r, context)
            if ctx.get("parent"):
                p = ctx["parent"]
                print(f"  --- родительский ---\n    [{p.get('date','')}] {p.get('author','')}: {p.get('text','')}")
            if ctx.get("replies"):
                print("  --- тред ---")
                for c in ctx["replies"]:
                    print(f"    [{c.get('date','')}] {c.get('author','')}: {c.get('text','')}")
        return
    # tg
    print(f"Чат: {r.get('chat_name','')}")
    print(f"Автор: {r.get('author','')}   Дата: {r.get('date','')}")
    print(f"Текст: {r.get('text','')}")
    print(f"Ссылка: {r.get('link','')}")
    if context > 0:
        ctx = _get_context(store, r, context)
        for label, items in (("до", ctx["before"]), ("после", ctx["after"])):
            if items:
                print(f"  --- {label} ---")
                for m in items:
                    print(_fmt_msg(m))


def run_search(
    query: str,
    k: int,
    collection: str,
    author: str | None,
    context: int,
    source: str | None,
    as_json: bool,
) -> None:
    emb = E5Embeddings()
    store = VectorStore(collection, dim=emb.dim)
    qvec = emb.embed_query(query)
    results = store.search(qvec, k=k, author=author, source=source)

    if as_json:
        out = []
        for r in results:
            item = dict(r)
            if context > 0:
                item["context"] = _get_context(store, r, context)
            out.append(item)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        store.close()
        return

    print(f"Запрос: {query!r} | коллекция={collection} | k={k}"
          + (f" | author={author}" if author else "")
          + (f" | source={source}" if source else ""))
    print("=" * 70)
    for i, r in enumerate(results, 1):
        _print_result(store, i, r, context)
    print("\n" + "=" * 70)
    print(f"Найдено: {len(results)}")
    store.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Семантический поиск по коллекции (Telegram + YouTube)")
    p.add_argument("query", help="Поисковый запрос")
    p.add_argument("-k", type=int, default=5, help="Сколько результатов вернуть")
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--author", default=None, help="Фильтр по автору (точное совпадение имени)")
    p.add_argument("--context", type=int, default=0, help="M соседних сообщений/чанков до и после")
    p.add_argument("--source", default=None, choices=["tg", "yt_comments", "yt_transcript"],
                   help="Фильтр по типу источника")
    p.add_argument("--json", action="store_true", help="Вывод в JSON")
    return p


def main() -> None:
    _reconfigure_stdout()
    args = build_parser().parse_args()
    run_search(
        query=args.query,
        k=args.k,
        collection=args.collection,
        author=args.author,
        context=args.context,
        source=args.source,
        as_json=args.json,
    )


if __name__ == "__main__":
    main()
