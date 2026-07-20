"""Ручной импорт транскрипции из скопированной панели YouTube «Показать расшифровку».

Когда субтитры недоступны через сеть (IP-лимит timedtext), можно скопировать расшифровку
руками из панели под видео и положить её в кэш — дальше пайплайн заэмбеддит как обычно.

Запуск:
    .\\.venv\\Scripts\\python.exe scripts\\import_transcript.py <URL_или_video_id> <файл.txt> [--lang ru]

Формат файла — как копируется из панели YouTube, оба варианта:
    метка и текст на разных строках        |   метка и текст в одной строке
        0:00                               |       0:00 Всем привет
        Всем привет                        |       0:03 сегодня поговорим
        0:03                               |
        сегодня поговорим                  |

Метки: mm:ss или h:mm:ss. Пишет transcript_<lang>.json в папку кэша видео
(data/youtube/...[video_id]/). Видео должно уже быть в кэше (info.json) — обычно так и есть
после первой попытки индексации; если нет, скрипт добёрет info через сеть.

После импорта заэмбеддить:
    .\\.venv\\Scripts\\python.exe -m src.index add-yt-transcript <URL> --collection <имя>
(или перезапустить пакетный ingest — он возьмёт транскрипцию из кэша, без сети).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.youtube import extract_video_id, fetch_video_info, find_video_dir, video_dir

# таймкод в начале строки: H:MM:SS или M:SS
_TS = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2})")

# метка времени прописью, которую YouTube клеит сразу за таймкодом БЕЗ пробела и без пробела
# перед текстом («0:5454 секундынаучпни»). Режем по точным словоформам (длинные раньше коротких),
# иначе жадный \w* съест первую букву текста («секундя» → «секунд»+«я»).
_LABEL = re.compile(
    r"^\s*(?:\d+\s*(?:секунду|секунды|секунда|секунд|минуту|минуты|минута|минут|часов|часа|час)\s*)+",
    re.IGNORECASE,
)


def parse_transcript(raw: str) -> list[dict]:
    """Разбирает скопированную панель YouTube в [{text, start, duration}].

    Понимает оба формата копирования панели:
      1) склеенный: «0:5454 секундытекст» — таймкод + метка прописью + текст в одной строке;
      2) построчный: строка с таймкодом «0:02», текст — на следующей строке.
    Строку без таймкода берём как текст только если реплика ещё не заполнена — так построчный
    текст (формат 2) подхватывается, а заголовки глав «Эпизод…» (формат 1, реплика уже с текстом)
    пропускаются. duration реплики = start следующей − start текущей (последняя — 0).
    """
    segments: list[tuple[int, str]] = []  # (start_sec, text)
    cur_start: int | None = None
    cur_text = ""
    filled = (
        False  # получила ли текущая реплика свой текст (защита от глав и лишних строк)
    )

    def flush() -> None:
        nonlocal cur_text, filled
        if cur_start is not None and cur_text.strip():
            segments.append((cur_start, cur_text.strip()))
        cur_text = ""
        filled = False

    for line in raw.splitlines():
        if not line.strip():
            continue
        m = _TS.match(line)
        if m:
            flush()
            h, mm, ss = m.group(1), m.group(2), m.group(3)
            cur_start = (int(h) if h else 0) * 3600 + int(mm) * 60 + int(ss)
            rest = _LABEL.sub(
                "", line[m.end() :], count=1
            ).strip()  # снять склеенную метку-прописью
            cur_text = rest
            filled = bool(rest)
        elif cur_start is not None and not filled:
            cur_text = line.strip()  # текст на отдельной строке (формат 2)
            filled = True
        # иначе (реплика уже заполнена) — заголовок главы / лишняя строка, пропускаем
    flush()

    snippets: list[dict] = []
    for i, (start, text) in enumerate(segments):
        nxt = segments[i + 1][0] if i + 1 < len(segments) else start
        snippets.append(
            {
                "text": text,
                "start": float(start),
                "duration": max(float(nxt - start), 0.0),
            }
        )
    return snippets


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="URL или video_id")
    ap.add_argument("txt", help="Путь к .txt с расшифровкой из панели YouTube")
    ap.add_argument("--lang", default="ru")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать существующий transcript_<lang>.json",
    )
    args = ap.parse_args()

    video_id = extract_video_id(args.url)
    raw = Path(args.txt).read_text(encoding="utf-8")
    snippets = parse_transcript(raw)
    if not snippets:
        print(
            "[error] Не удалось распарсить ни одной реплики с меткой времени. "
            "Проверь, что копировал из панели «Показать расшифровку» (с таймкодами)."
        )
        sys.exit(1)

    d = find_video_dir(video_id)
    if d is None:
        info = fetch_video_info(video_id)  # добор через сеть, если видео ещё не в кэше
        d = video_dir(video_id, info["channel"], info["title"])

    out = d / f"transcript_{args.lang}.json"
    if out.exists() and out.stat().st_size > 2 and not args.force:
        print(
            f"[warn] {out.name} уже есть и не пуст ({out.stat().st_size} байт). "
            f"Добавь --force, чтобы перезаписать."
        )
        sys.exit(1)

    out.write_text(json.dumps(snippets, ensure_ascii=False), encoding="utf-8")
    total = snippets[-1]["start"] if snippets else 0
    print(
        f"OK: {len(snippets)} реплик, до {int(total // 60)}:{int(total % 60):02d}  →  {out}"
    )
    print(
        f"Теперь заэмбеддить: .\\.venv\\Scripts\\python.exe -m src.index add-yt-transcript "
        f'"{args.url}" --collection <имя>'
    )


if __name__ == "__main__":
    main()
