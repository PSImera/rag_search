# rag_search

Локальный семантический (RAG) поиск по личным архивам: Telegram-чаты, комментарии и транскрипции YouTube-видео — всё в одном индексе, с прямыми ссылками на оригинал (сообщение в чате, комментарий под видео, таймкод в ролике). Работает полностью офлайн после выгрузки данных: эмбеддинги считаются локально на GPU, платные API не используются.

## Возможности

- **Именованные коллекции** — изолированные индексы под разные темы (`store/<имя>/`).
- **Три типа источников** в одной коллекции:
  - `tg` — сообщения из JSON-экспорта Telegram Desktop;
  - `yt_comments` — комментарии YouTube-видео (через yt-dlp, без API-ключа);
  - `yt_transcript` — транскрипции видео, склеенные в чанки с таймкодами.
- **Контекст выдачи** — соседние сообщения / тред комментариев / соседние фрагменты транскрипции из локального SQLite.
- **Идемпотентность** — повторный запуск индексации ничего не дублирует и работает как resume; выгрузки YouTube кэшируются в `data/youtube/` и повторно сеть не трогают.

## Структура проекта

```
src/        код (config.py — все пути и константы)
scripts/    пакетные скрипты (ingest_videos.py)
data/       первоисточники (не в git)
  tg_chats/   JSON-экспорты Telegram Desktop
  youtube/    кэш выгрузок: <Канал> - <Название> [video_id]/
docs/       списки ссылок на видео и прочие документы (не в git)
logs/       логи и отчёты индексации (не в git)
store/      коллекции: index.faiss, docstore.pkl, raw.sqlite, meta.json (не в git)
```

## Установка

Требования: Windows, Python 3.10+, NVIDIA GPU (тестировалось на RTX 3070 Ti; на CPU тоже работает, но эмбеддинг больших чатов будет долгим).

```powershell
cd D:\My_Code\rag_search
python -m venv .venv
# GPU-сборка torch (для Python 3.10–3.12):
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
# для Python 3.13 колёс под cu121 нет — брать cu126:
# .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Модель эмбеддингов `intfloat/multilingual-e5-large` (~2.2 ГБ) скачается в `~/.cache/huggingface` при первом запуске, дальше работает офлайн.

Все команды запускаются через python из `.venv`:

```powershell
.\.venv\Scripts\python.exe -m src.index list
```

## Использование

### 1. Добавить Telegram-чат

Экспортируйте чат из Telegram Desktop (⋮ → Export chat history → формат JSON, медиа не нужны), положите JSON в `data/tg_chats/` и запустите:

```powershell
.\.venv\Scripts\python.exe -m src.index add-tg-chat "data\tg_chats\ChatExport.json" --collection mytopic --min-chars 80
```

- `--collection` — имя коллекции; новой темы ещё нет — папка создастся сама. Указывайте всегда явно.
- `--min-chars N` — эмбеддятся сообщения длиной ≥N (дефолт 80); короткие всё равно попадают в SQLite и видны в контексте выдачи.
- `--username имя` — для чатов с публичным username (ссылки вида `t.me/имя/<id>`); по умолчанию ссылки для приватных супергрупп `t.me/c/<chat_id>/<id>`.

### 2. Добавить YouTube-видео

Комментарии и/или транскрипция, по одному видео:

```powershell
.\.venv\Scripts\python.exe -m src.index add-yt-comments "https://youtu.be/KjhI4vDfMdY" --collection mytopic
.\.venv\Scripts\python.exe -m src.index add-yt-transcript "https://youtu.be/KjhI4vDfMdY" --collection mytopic --lang ru
```

Пакетно — список ссылок в файле (по одной в строке, `#` — комментарий), удобно держать в `docs/`:

```powershell
.\.venv\Scripts\python.exe scripts\ingest_videos.py docs\videos.md --collection mytopic
```

Сводная таблица выводится в конце, отчёт пишется в `logs/ingest_report.json`. Полный лог можно сохранить перенаправлением: `... | Tee-Object logs\ingest_log.txt`.

**Не запускайте два процесса индексации одновременно** — они пишут в один index.faiss и портят его.

### 3. Искать

```powershell
.\.venv\Scripts\python.exe -m src.search "наука единственный способ познания" -k 10 --collection mytopic --context 3
```

- `-k N` — сколько результатов (дефолт 5).
- `--context M` — M соседей до/после (сообщения чата / тред комментариев / соседние фрагменты).
- `--source tg|yt_comments|yt_transcript` — фильтр по типу источника.
- `--author "Имя"` — фильтр по автору (для YouTube — @handle).
- `--json` — машиночитаемый вывод.

Каждый результат содержит прямую ссылку: на сообщение в Telegram, на конкретный комментарий под видео или на таймкод в ролике.

### Управление коллекциями

- Список со статистикой: `.\.venv\Scripts\python.exe -m src.index list`
- Удалить коллекцию: удалить папку `store/<имя>/`.
- Прервали индексацию — просто повторите команду: уже добавленное пропустится.

## Частые проблемы

- `ModuleNotFoundError` — запущен глобальный python; используйте `.\.venv\Scripts\python.exe`.
- Кракозябры в PowerShell — `[Console]::OutputEncoding = [Text.Encoding]::UTF8`.
- yt-dlp падает на всех видео — YouTube поменял API: `.\.venv\Scripts\python.exe -m pip install -U yt-dlp`.
- `faiss.read_index` падает с ошибкой `size >= 0 && size < (1<<40)` — index.faiss повреждён (обычно после параллельных запусков индексации). Удалите `store/<имя>/index.faiss` и `docstore.pkl` (raw.sqlite и кэш `data/` можно оставить) и переиндексируйте источники — сеть не понадобится, всё возьмётся из кэша.
