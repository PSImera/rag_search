# rag_search

Локальный семантический (RAG) поиск по личным архивам: Telegram-чаты, комментарии и транскрипции YouTube-видео — всё в одном индексе, с прямыми ссылками на оригинал (сообщение в чате, комментарий под видео, таймкод в ролике). Работает полностью офлайн после выгрузки данных: эмбеддинги считаются локально на GPU, платные API не используются.

## Возможности

- **Именованные коллекции** — изолированные индексы под разные темы (`store/<имя>/`).
- **Три типа источников** в одной коллекции:
  - `tg` — сообщения из JSON-экспорта Telegram Desktop;
  - `yt_comments` — комментарии YouTube-видео (через yt-dlp, без API-ключа);
  - `yt_transcript` — транскрипции видео (субтитры YouTube через yt-dlp, а при их отсутствии — локальное распознавание аудио через Whisper), склеенные в чанки с таймкодами.
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

Субтитры забираются через yt-dlp. Если субтитров нет или они кривые (частая беда автосубтитров), добавьте `--asr` — субтитры YouTube игнорируются полностью, а транскрипт строится распознаванием аудио локально через Whisper (`faster-whisper`, GPU):

```powershell
.\.venv\Scripts\python.exe -m src.index add-yt-transcript "https://youtu.be/KjhI4vDfMdY" --collection mytopic --lang ru --asr
```

- `--asr` качает аудио во временную папку и удаляет его сразу после распознавания (в кэше остаётся только готовый текст транскрипции). Уже готовый непустой транскрипт переиспользуется — чтобы перегнать заново, удалите папку видео в `data/youtube/`.
- `--asr-model` — модель распознавания (дефолт `large-v3`; полегче/побыстрее — `medium`, `distil-large-v3`).

Пакетно — список ссылок в файле (по одной в строке, `#` — комментарий), удобно держать в `docs/`. Ссылка на плейлист (`youtube.com/playlist?list=...`) разворачивается во все видео плейлиста автоматически:

```powershell
.\.venv\Scripts\python.exe scripts\ingest_videos.py docs\videos.md --collection mytopic
# --asr: субтитры YouTube игнорируются, транскрипт всех видео строится из аудио (Whisper):
.\.venv\Scripts\python.exe scripts\ingest_videos.py docs\videos.md --collection mytopic --asr
```

Сводная таблица выводится в конце, отчёт пишется в `logs/ingest_report.json`. Полный лог можно сохранить перенаправлением: `... | Tee-Object logs\ingest_log.txt`.

**Не запускайте два процесса индексации одновременно** — они пишут в один index.faiss и портят его.

### 3. Искать

```powershell
.\.venv\Scripts\python.exe -m src.search "наука единственный способ познания" -k 10 --collection mytopic --context 3
```

- `-k N` — сколько результатов (дефолт 5).
- `--context M` — M соседей до/после (сообщения чата / тред комментариев / соседние фрагменты).
- `--json` — машиночитаемый вывод.

Фильтры по метаданным (применяются к уже готовому индексу — эмбеддинги не пересчитываются, переиндексация не нужна; все фильтры комбинируются по И):

- `--source tg|yt_comments|yt_transcript` — только один тип источника.
- `--author "Имя"` — точное совпадение имени автора (для YouTube — @handle).
- `--chat X` — только этот Telegram-чат: точный `chat_id` **или** подстрока названия чата.
- `--video X` — только это YouTube-видео: точный `video_id` **или** подстрока названия видео.
- `--channel X` — только этот YouTube-канал: подстрока названия канала.
- `--filter "поле=значение"` — универсальный фильтр по любому полю метаданных: `=` точное совпадение, `~` вхождение подстроки (оба регистронезависимы). Можно повторять — условия складываются по И.

```powershell
# только транскрипции конкретного канала
.\.venv\Scripts\python.exe -m src.search "научный метод" --collection mytopic --channel "Панчин" --source yt_transcript

# конкретное видео (по id или части названия) + универсальный фильтр по языку субтитров
.\.venv\Scripts\python.exe -m src.search "научный метод" --collection mytopic --video KjhI4vDfMdY --filter "lang=ru"
```

Готовый список всех источников коллекции (id и названия чатов, каналов, роликов — чтобы было что подставлять в `--chat`/`--video`/`--channel`) выгружается в `docs/sources_<коллекция>.md`:

```powershell
.\.venv\Scripts\python.exe scripts\list_sources.py --collection mytopic
```

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
