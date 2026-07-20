"""Выгружает полный список источников коллекции в markdown-файл (docs/).

Запуск:
    .\\.venv\\Scripts\\python.exe scripts\\list_sources.py --collection scientism

Собирает из docstore (метаданные чанков) все ТГ-чаты, YouTube-каналы и ролики
вместе с их идентификаторами и названиями — удобная шпаргалка, откуда брать
значения для фильтров поиска `--chat` / `--video` / `--channel`.

По умолчанию пишет в docs/sources_<коллекция>.md; путь можно задать через --out.
GPU и сеть не задействуются: читается только локальный docstore коллекции.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_COLLECTION, PROJECT_ROOT
from src.vectorstore import VectorStore


def collect(collection: str) -> dict:
    store = VectorStore(collection)
    tg_chats: dict[str, dict] = {}
    videos: dict[str, dict] = {}
    try:
        for m in store.docstore.values():
            src = m.get("source", "tg")
            if src == "tg":
                cid = str(m.get("chat_id", ""))
                rec = tg_chats.setdefault(
                    cid, {"chat_name": m.get("chat_name") or "", "chunks": 0}
                )
                rec["chunks"] += 1
                if not rec["chat_name"] and m.get("chat_name"):
                    rec["chat_name"] = m["chat_name"]
            else:  # yt_comments / yt_transcript
                vid = str(m.get("video_id", ""))
                rec = videos.setdefault(
                    vid,
                    {
                        "video_title": m.get("video_title") or "",
                        "channel": m.get("channel") or "",
                        "yt_comments": 0,
                        "yt_transcript": 0,
                    },
                )
                rec[src] = rec.get(src, 0) + 1
                if not rec["video_title"] and m.get("video_title"):
                    rec["video_title"] = m["video_title"]
                if not rec["channel"] and m.get("channel"):
                    rec["channel"] = m["channel"]
    finally:
        store.close()
    return {"tg_chats": tg_chats, "videos": videos}


def render(collection: str, data: dict) -> str:
    tg_chats = data["tg_chats"]
    videos = data["videos"]

    lines: list[str] = []
    lines.append(f"# Источники коллекции `{collection}`")
    lines.append("")
    lines.append(
        "Автогенерируется из docstore: "
        "`.\\.venv\\Scripts\\python.exe scripts\\list_sources.py --collection "
        f"{collection}`. Значения из колонок можно подставлять в фильтры поиска "
        "(`--chat`, `--video`, `--channel`, `--author`)."
    )
    lines.append("")

    # --- Telegram ---
    lines.append("## Telegram-чаты")
    lines.append("")
    if tg_chats:
        lines.append(
            'Фильтр: `--chat <chat_id>` (точно) или `--chat "<часть названия>"` (подстрока).'
        )
        lines.append("")
        lines.append("| chat_id | Название | Проиндексировано чанков |")
        lines.append("|---|---|---|")
        for cid, rec in sorted(tg_chats.items(), key=lambda kv: -kv[1]["chunks"]):
            lines.append(f"| `{cid}` | {rec['chat_name']} | {rec['chunks']} |")
    else:
        lines.append("_нет_")
    lines.append("")

    # --- YouTube ---
    lines.append("## YouTube")
    lines.append("")
    if videos:
        # каналы
        channels: dict[str, int] = {}
        for rec in videos.values():
            channels[rec["channel"]] = channels.get(rec["channel"], 0) + 1
        lines.append("### Каналы")
        lines.append("")
        lines.append('Фильтр: `--channel "<часть названия канала>"` (подстрока).')
        lines.append("")
        lines.append("| Канал | Роликов |")
        lines.append("|---|---|")
        for ch, n in sorted(channels.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"| {ch or '—'} | {n} |")
        lines.append("")

        # ролики
        lines.append("### Ролики")
        lines.append("")
        lines.append(
            'Фильтр: `--video <video_id>` (точно) или `--video "<часть названия>"` (подстрока). '
            "Столбцы «комм.»/«транскр.» — число проиндексированных чанков каждого типа "
            "(0 = источник не добавлен)."
        )
        lines.append("")
        lines.append("| video_id | Канал | Название | комм. | транскр. |")
        lines.append("|---|---|---|---|---|")
        for vid, rec in sorted(
            videos.items(), key=lambda kv: (kv[1]["channel"], kv[1]["video_title"])
        ):
            lines.append(
                f"| `{vid}` | {rec['channel']} | {rec['video_title']} "
                f"| {rec.get('yt_comments', 0)} | {rec.get('yt_transcript', 0)} |"
            )
    else:
        lines.append("_нет_")
    lines.append("")

    lines.append("---")
    lines.append(f"Итого: чатов ТГ — {len(tg_chats)}, роликов YouTube — {len(videos)}.")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Список источников коллекции -> markdown в docs/"
    )
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument(
        "--out",
        default=None,
        help="Куда писать (по умолчанию docs/sources_<коллекция>.md)",
    )
    return p


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = build_parser().parse_args()
    data = collect(args.collection)
    text = render(args.collection, data)

    out = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT / "docs" / f"sources_{args.collection}.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Записано: {out}")
    print(f"  ТГ-чатов: {len(data['tg_chats'])}, роликов: {len(data['videos'])}")


if __name__ == "__main__":
    main()
