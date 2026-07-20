from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import ijson


def extract_text(raw_text: Any) -> str:
    """Склеивает поле text экспорта Telegram.

    text бывает: строкой, либо массивом фрагментов (строки + объекты {"type","text"}).
    """
    if raw_text is None:
        return ""
    if isinstance(raw_text, str):
        return raw_text
    if isinstance(raw_text, list):
        parts: list[str] = []
        for frag in raw_text:
            if isinstance(frag, str):
                parts.append(frag)
            elif isinstance(frag, dict):
                parts.append(str(frag.get("text", "")))
        return "".join(parts)
    return str(raw_text)


def read_chat_header(json_path: Path) -> dict[str, Any]:
    """Читает шапку экспорта (name, type, id) без загрузки всех сообщений."""
    header: dict[str, Any] = {}
    with json_path.open("r", encoding="utf-8") as f:
        for key in ("name", "type", "id"):
            f.seek(0)
            try:
                header[key] = next(ijson.items(f, key))
            except StopIteration:
                header[key] = None
    return header


def iter_messages(json_path: Path) -> Iterator[dict[str, Any]]:
    """Потоково выдаёт нормализованные сообщения из экспорта.

    Пропускает служебные сообщения (type != "message") и пустые тексты.
    Не грузит весь 166 МБ файл в память — использует ijson.
    """
    with json_path.open("r", encoding="utf-8") as f:
        for msg in ijson.items(f, "messages.item"):
            if msg.get("type") != "message":
                continue
            text = extract_text(msg.get("text")).strip()
            if not text:
                continue
            yield {
                "message_id": int(msg["id"]),
                "date": msg.get("date", ""),
                "author": msg.get("from") or "",
                "author_id": msg.get("from_id") or "",
                "reply_to_message_id": msg.get("reply_to_message_id"),
                "text": text,
            }


def make_link(
    link_template: str, chat_id: Any, username: str | None, message_id: int
) -> str:
    """Формирует прямую ссылку на сообщение.

    Если у чата есть публичный username — t.me/<username>/<message_id>.
    Иначе применяется link_template (дефолт t.me/c/{chat_id}/{message_id}).
    chat_id в экспортах Telegram Desktop уже без префикса -100.
    """
    if username:
        return f"https://t.me/{username}/{message_id}"
    return link_template.format(chat_id=chat_id, message_id=message_id)
