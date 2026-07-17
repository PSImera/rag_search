from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable


class RawStore:
    """SQLite-хранилище ВСЕХ сырых сообщений (в т.ч. коротких, не попавших в индекс).

    Нужно для --context: достать соседние по message_id реплики без обращения к Telegram.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                chat_id     TEXT NOT NULL,
                message_id  INTEGER NOT NULL,
                author      TEXT,
                author_id   TEXT,
                date        TEXT,
                reply_to    INTEGER,
                text        TEXT,
                link        TEXT,
                PRIMARY KEY (chat_id, message_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS yt_comments (
                video_id    TEXT NOT NULL,
                comment_id  TEXT NOT NULL,
                parent_id   TEXT,
                author      TEXT,
                author_id   TEXT,
                date        TEXT,
                likes       INTEGER,
                text        TEXT,
                link        TEXT,
                PRIMARY KEY (video_id, comment_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS yt_chunks (
                video_id    TEXT NOT NULL,
                start_sec   INTEGER NOT NULL,
                end_sec     INTEGER,
                text        TEXT,
                link        TEXT,
                video_title TEXT,
                channel     TEXT,
                lang        TEXT,
                PRIMARY KEY (video_id, start_sec)
            )
            """
        )
        self.conn.commit()

    def upsert_many(self, rows: Iterable[dict[str, Any]]) -> None:
        self.conn.executemany(
            """
            INSERT INTO messages (chat_id, message_id, author, author_id, date, reply_to, text, link)
            VALUES (:chat_id, :message_id, :author, :author_id, :date, :reply_to, :text, :link)
            ON CONFLICT(chat_id, message_id) DO UPDATE SET
                author=excluded.author, author_id=excluded.author_id, date=excluded.date,
                reply_to=excluded.reply_to, text=excluded.text, link=excluded.link
            """,
            list(rows),
        )
        self.conn.commit()

    def get_context(self, chat_id: str, message_id: int, m: int) -> dict[str, list[dict[str, Any]]]:
        """Возвращает до m сообщений до и после указанного (по message_id) в том же чате."""
        before = self.conn.execute(
            """
            SELECT chat_id, message_id, author, author_id, date, reply_to, text, link
            FROM messages WHERE chat_id=? AND message_id < ?
            ORDER BY message_id DESC LIMIT ?
            """,
            (chat_id, message_id, m),
        ).fetchall()
        after = self.conn.execute(
            """
            SELECT chat_id, message_id, author, author_id, date, reply_to, text, link
            FROM messages WHERE chat_id=? AND message_id > ?
            ORDER BY message_id ASC LIMIT ?
            """,
            (chat_id, message_id, m),
        ).fetchall()
        cols = ["chat_id", "message_id", "author", "author_id", "date", "reply_to", "text", "link"]
        to_dicts = lambda rows: [dict(zip(cols, r)) for r in rows]
        return {"before": list(reversed(to_dicts(before))), "after": to_dicts(after)}

    def get_one(self, chat_id: str, message_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT chat_id, message_id, author, author_id, date, reply_to, text, link
            FROM messages WHERE chat_id=? AND message_id=?
            """,
            (chat_id, message_id),
        ).fetchone()
        if row is None:
            return None
        cols = ["chat_id", "message_id", "author", "author_id", "date", "reply_to", "text", "link"]
        return dict(zip(cols, row))

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

    # ---------- YouTube comments ----------

    def upsert_yt_comments(self, rows: Iterable[dict[str, Any]]) -> None:
        self.conn.executemany(
            """
            INSERT INTO yt_comments (video_id, comment_id, parent_id, author, author_id, date, likes, text, link)
            VALUES (:video_id, :comment_id, :parent_id, :author, :author_id, :date, :likes, :text, :link)
            ON CONFLICT(video_id, comment_id) DO UPDATE SET
                parent_id=excluded.parent_id, author=excluded.author, author_id=excluded.author_id,
                date=excluded.date, likes=excluded.likes, text=excluded.text, link=excluded.link
            """,
            list(rows),
        )
        self.conn.commit()

    def get_comment_thread(self, video_id: str, comment_id: str, m: int) -> dict[str, Any]:
        """Контекст комментария: родитель + до m ответов того же треда."""
        cols = ["video_id", "comment_id", "parent_id", "author", "author_id", "date", "likes", "text", "link"]
        q = "SELECT video_id, comment_id, parent_id, author, author_id, date, likes, text, link FROM yt_comments WHERE video_id=? AND comment_id=?"

        row = self.conn.execute(q, (video_id, comment_id)).fetchone()
        me = dict(zip(cols, row)) if row else None
        parent = None
        thread_root = comment_id
        if me and me["parent_id"]:
            thread_root = me["parent_id"]
            prow = self.conn.execute(q, (video_id, me["parent_id"])).fetchone()
            parent = dict(zip(cols, prow)) if prow else None
        replies = self.conn.execute(
            """
            SELECT video_id, comment_id, parent_id, author, author_id, date, likes, text, link
            FROM yt_comments WHERE video_id=? AND parent_id=? AND comment_id != ?
            ORDER BY date ASC LIMIT ?
            """,
            (video_id, thread_root, comment_id, m),
        ).fetchall()
        return {"parent": parent, "replies": [dict(zip(cols, r)) for r in replies]}

    # ---------- YouTube transcript chunks ----------

    def upsert_yt_chunks(self, rows: Iterable[dict[str, Any]]) -> None:
        self.conn.executemany(
            """
            INSERT INTO yt_chunks (video_id, start_sec, end_sec, text, link, video_title, channel, lang)
            VALUES (:video_id, :start_sec, :end_sec, :text, :link, :video_title, :channel, :lang)
            ON CONFLICT(video_id, start_sec) DO UPDATE SET
                end_sec=excluded.end_sec, text=excluded.text, link=excluded.link,
                video_title=excluded.video_title, channel=excluded.channel, lang=excluded.lang
            """,
            list(rows),
        )
        self.conn.commit()

    def get_chunk_context(self, video_id: str, start_sec: int, m: int) -> dict[str, list[dict[str, Any]]]:
        """Соседние чанки транскрипции по времени."""
        cols = ["video_id", "start_sec", "end_sec", "text", "link", "video_title", "channel", "lang"]
        before = self.conn.execute(
            """
            SELECT video_id, start_sec, end_sec, text, link, video_title, channel, lang
            FROM yt_chunks WHERE video_id=? AND start_sec < ? ORDER BY start_sec DESC LIMIT ?
            """,
            (video_id, start_sec, m),
        ).fetchall()
        after = self.conn.execute(
            """
            SELECT video_id, start_sec, end_sec, text, link, video_title, channel, lang
            FROM yt_chunks WHERE video_id=? AND start_sec > ? ORDER BY start_sec ASC LIMIT ?
            """,
            (video_id, start_sec, m),
        ).fetchall()
        to_dicts = lambda rows: [dict(zip(cols, r)) for r in rows]
        return {"before": list(reversed(to_dicts(before))), "after": to_dicts(after)}

    # ---------- Удаление источников ----------

    def delete_chat(self, chat_id: str) -> int:
        """Удаляет все сырые сообщения чата. Возвращает число удалённых строк."""
        n = self.conn.execute("DELETE FROM messages WHERE chat_id=?", (str(chat_id),)).rowcount
        self.conn.commit()
        return n

    def delete_video(self, video_id: str) -> tuple[int, int]:
        """Удаляет сырые комментарии и чанки транскрипции видео. Возвращает (комментов, чанков)."""
        c = self.conn.execute("DELETE FROM yt_comments WHERE video_id=?", (str(video_id),)).rowcount
        ch = self.conn.execute("DELETE FROM yt_chunks WHERE video_id=?", (str(video_id),)).rowcount
        self.conn.commit()
        return c, ch

    def stats(self) -> dict[str, Any]:
        """Статистика по источникам для index list."""
        out: dict[str, Any] = {}
        out["tg_messages"] = self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        out["tg_chats"] = [
            {"chat_id": r[0], "messages": r[1]}
            for r in self.conn.execute("SELECT chat_id, COUNT(*) FROM messages GROUP BY chat_id").fetchall()
        ]
        out["yt_comments"] = self.conn.execute("SELECT COUNT(*) FROM yt_comments").fetchone()[0]
        out["yt_videos_comments"] = [
            {"video_id": r[0], "comments": r[1]}
            for r in self.conn.execute("SELECT video_id, COUNT(*) FROM yt_comments GROUP BY video_id").fetchall()
        ]
        out["yt_chunks"] = self.conn.execute("SELECT COUNT(*) FROM yt_chunks").fetchone()[0]
        out["yt_videos_chunks"] = [
            {"video_id": r[0], "title": r[2], "chunks": r[1]}
            for r in self.conn.execute(
                "SELECT video_id, COUNT(*), MAX(video_title) FROM yt_chunks GROUP BY video_id"
            ).fetchall()
        ]
        return out

    def close(self) -> None:
        self.conn.close()
