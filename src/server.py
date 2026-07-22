"""Локальный HTTP-сервер для веб-фронтенда поиска.

Запуск:
    .\\.venv\\Scripts\\python.exe -m src.server
    .\\.venv\\Scripts\\python.exe -m src.server --host 127.0.0.1 --port 8000

Отдаёт статику из frontend/ и JSON-API поверх того же движка, что и CLI-поиск:
    GET /api/collections            — список коллекций со статистикой (без GPU);
    GET /api/sources?collection=X   — источники коллекции для фильтров (чаты,
                                       каналы, видео, авторы) — читает docstore;
    POST /api/search {json}         — семантический поиск, тело как у CLI-опций.

Модель эмбеддингов грузится ЛЕНИВО при первом поиске и живёт до конца процесса;
VectorStore кэшируется по имени коллекции. Поиск сериализуется локом (одна GPU,
faiss-индекс на процесс). Зависимостей сверх requirements.txt не добавляет —
только stdlib http.server.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.config import DEFAULT_COLLECTION, STORE_ROOT
from src.search import execute_search
from src.vectorstore import VectorStore

FRONTEND_ROOT = Path(__file__).resolve().parent.parent / "frontend"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".png": "image/png",
}


class Engine:
    """Кэш модели и коллекций на процесс.

    Весь поиск исполняется в ОДНОМ выделенном потоке (`ThreadPoolExecutor` на 1
    воркера): это и сериализует доступ к единственной GPU/faiss-индексу, и держит
    все обращения к SQLite (`RawStore`) в том же потоке, где соединение создано —
    иначе ThreadingHTTPServer раздаёт запросы по разным потокам и sqlite падает
    с 'objects created in a thread can only be used in that same thread'."""

    def __init__(self) -> None:
        self._emb = None
        self._stores: dict[str, VectorStore] = {}
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-search")

    def _get_emb(self):
        if self._emb is None:
            from src.embeddings import E5Embeddings  # тяжёлый импорт — только при поиске

            self._emb = E5Embeddings()
        return self._emb

    def _get_store(self, collection: str) -> VectorStore:
        store = self._stores.get(collection)
        if store is None:
            store = VectorStore(collection, dim=self._get_emb().dim)
            self._stores[collection] = store
        return store

    def search(self, params: dict) -> list[dict]:
        """Валидирует запрос и исполняет его в выделенном потоке (см. класс)."""
        query = (params.get("query") or "").strip()
        if not query:
            raise ValueError("пустой запрос")
        collection = params.get("collection") or DEFAULT_COLLECTION
        if not (STORE_ROOT / collection).is_dir():
            raise ValueError(f"нет коллекции {collection!r}")
        return self._pool.submit(self._search, query, collection, params).result()

    def _search(self, query: str, collection: str, params: dict) -> list[dict]:
        def _clean(v):
            v = (v or "").strip() if isinstance(v, str) else v
            return v or None

        filters = params.get("filters") or None
        if isinstance(filters, str):
            filters = [f for f in (x.strip() for x in filters.splitlines()) if f]

        store = self._get_store(collection)
        return execute_search(
            store=store,
            emb=self._get_emb(),
            query=query,
            k=int(params.get("k") or 5),
            context=int(params.get("context") or 0),
            source=_clean(params.get("source")),
            author=_clean(params.get("author")),
            chat=_clean(params.get("chat")),
            video=_clean(params.get("video")),
            channel=_clean(params.get("channel")),
            filters=filters,
        )


def list_collections() -> list[dict]:
    """Имена коллекций + краткая статистика из meta.json (без загрузки индекса/GPU)."""
    out: list[dict] = []
    if not STORE_ROOT.exists():
        return out
    for d in sorted(STORE_ROOT.iterdir()):
        if not d.is_dir() or not (d / "index.faiss").exists():
            continue
        vectors = None
        meta_file = d / "meta.json"
        if meta_file.exists():
            try:
                vectors = json.loads(meta_file.read_text(encoding="utf-8")).get("vectors")
            except Exception:
                vectors = None
        out.append({"name": d.name, "vectors": vectors})
    return out


def list_sources(collection: str) -> dict:
    """Чаты, каналы, видео и авторы коллекции для выпадающих фильтров.

    Читает docstore (без GPU и сети). Авторы ограничены сверху, чтобы не раздувать
    ответ на больших чатах — точный фильтр по автору всё равно доступен вводом."""
    store = VectorStore(collection)
    tg_chats: dict[str, dict] = {}
    channels: dict[str, int] = {}
    videos: dict[str, dict] = {}
    try:
        for m in store.docstore.values():
            src = m.get("source", "tg")
            if src == "tg":
                cid = str(m.get("chat_id", ""))
                rec = tg_chats.setdefault(
                    cid, {"chat_id": cid, "chat_name": m.get("chat_name") or "", "count": 0}
                )
                rec["count"] += 1
                if not rec["chat_name"] and m.get("chat_name"):
                    rec["chat_name"] = m["chat_name"]
            else:
                vid = str(m.get("video_id", ""))
                ch = m.get("channel") or ""
                rec = videos.setdefault(
                    vid,
                    {"video_id": vid, "video_title": m.get("video_title") or "",
                     "channel": ch, "count": 0},
                )
                rec["count"] += 1
                if not rec["video_title"] and m.get("video_title"):
                    rec["video_title"] = m["video_title"]
                if not rec["channel"] and ch:
                    rec["channel"] = ch
                if ch:
                    channels[ch] = channels.get(ch, 0) + 1
    finally:
        store.close()

    return {
        "tg_chats": sorted(tg_chats.values(), key=lambda r: -r["count"]),
        "channels": [
            {"channel": c, "count": n}
            for c, n in sorted(channels.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "videos": sorted(
            videos.values(), key=lambda r: (r["channel"], r["video_title"])
        ),
    }


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # тише в консоли
            sys.stderr.write("  %s - %s\n" % (self.address_string(), format % args))

        def _send_json(self, obj, status: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error_json(self, status: int, message: str) -> None:
            self._send_json({"error": message}, status=status)

        # ---- API ----
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/api/collections":
                try:
                    self._send_json({"collections": list_collections()})
                except Exception as e:  # noqa: BLE001
                    self._send_error_json(500, str(e))
                return
            if path == "/api/sources":
                qs = parse_qs(parsed.query)
                collection = (qs.get("collection") or [DEFAULT_COLLECTION])[0]
                if not (STORE_ROOT / collection).is_dir():
                    self._send_error_json(404, f"нет коллекции {collection!r}")
                    return
                try:
                    self._send_json(list_sources(collection))
                except Exception as e:  # noqa: BLE001
                    self._send_error_json(500, str(e))
                return
            self._serve_static(path)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/search":
                self._send_error_json(404, "не найдено")
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send_error_json(400, "некорректный JSON")
                return
            try:
                results = engine.search(payload)
                self._send_json({"results": results, "count": len(results)})
            except ValueError as e:
                self._send_error_json(400, str(e))
            except Exception as e:  # noqa: BLE001
                self._send_error_json(500, str(e))

        # ---- статика ----
        def _serve_static(self, path: str) -> None:
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            target = (FRONTEND_ROOT / rel).resolve()
            try:
                target.relative_to(FRONTEND_ROOT.resolve())
            except ValueError:
                self._send_error_json(403, "запрещено")
                return
            if not target.is_file():
                self._send_error_json(404, "не найдено")
                return
            data = target.read_bytes()
            self.send_response(200)
            self.send_header(
                "Content-Type",
                _CONTENT_TYPES.get(target.suffix, "application/octet-stream"),
            )
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description="Веб-сервер поиска (frontend + JSON API)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    engine = Engine()
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    url = f"http://{args.host}:{args.port}/"
    print(f"RAG-поиск: {url}")
    print("Модель эмбеддингов загрузится при первом запросе. Ctrl+C — остановить.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
