# RAG-поиск: Telegram-чаты + YouTube (комментарии и транскрипции)

Инструкции для агента. Установка и подробный пользовательский гайд — в [README.md](README.md).

Семантический поиск по именованным коллекциям. Три типа источников в одном индексе:
`tg` (сообщения Telegram), `yt_comments` (комментарии YouTube), `yt_transcript` (фрагменты транскрипций).

## Структура проекта

```
src/        код (config.py — все пути и константы)
scripts/    пакетные скрипты (ingest_videos.py)
data/       первоисточники: tg_chats/ (JSON-экспорты), youtube/ (кэш по видео)
docs/       списки ссылок на видео и прочие документы (video_for_seach_scientism.md)
logs/       ingest_log.txt, ingest_report.json
store/      коллекции: <имя>/(index.faiss, docstore.pkl, raw.sqlite, meta.json)
```

`data/`, `docs/`, `logs/`, `store/` в git не отслеживаются — только код.

## Стек

- Эмбеддинги: `intfloat/multilingual-e5-large` — локально (sentence-transformers, CUDA автоматически). Платные API не используются. Модель в кеше `~/.cache/huggingface`, работает офлайн.
- Вектора: FAISS `IndexFlatIP + IndexIDMap2`, нормированные вектора (косинус).
- Сырые данные для контекста: SQLite в каждой коллекции (таблицы `messages`, `yt_comments`, `yt_chunks`).
- Кэш YouTube: `data/youtube/<Канал> - <Название> [video_id]/` — `info.json`, `comments.json`, `transcript_<lang>.json`. Повторная индексация берёт данные из кэша, сеть не трогает.

## Окружение

Всегда запускай через python из `.venv` (глобальный python пакетов не имеет):

```powershell
cd D:\My_Code\rag_search
.\.venv\Scripts\python.exe -m src.index list
```

Пересоздание окружения — см. README.md (только если .venv сломан/отсутствует).

## Коллекции

Коллекция = папка `store/<имя>/`. Полностью изолированы.

- Новая тема → новая коллекция: просто передай новое имя в `--collection` при первой индексации — папка создастся сама.
- Удалить коллекцию целиком: удалить папку `store/<имя>/`.
- Удалить один источник из коллекции (без пересоздания, без GPU): `delete-source`. Чистит FAISS-индекс + docstore + SQLite согласованно; эмбеддинги не пересчитываются.
  - Видео: `.\.venv\Scripts\python.exe -m src.index delete-source --collection scientism --video KjhI4vDfMdY`
  - ТГ-чат: `.\.venv\Scripts\python.exe -m src.index delete-source --collection scientism --chat 1545461926`
  - Спрашивает подтверждение; `--yes` — без вопроса. Чистка «сирот» после смены схемы чанков: `delete-source` нужного видео/чата, затем повторная индексация его же — соберётся заново по актуальной схеме.
- Список коллекций со статистикой: `.\.venv\Scripts\python.exe -m src.index list`
- Существующая коллекция: `scientism` — два ТГ-чата (Kortik Chat, Science Inquisition Comments, ~294k сообщений) + видео по сциентизму (комментарии + транскрипции).

ПРАВИЛО: во всех командах указывай `--collection` явно (дефолт `default` — почти наверняка не то, что нужно).

## Добавление источников (всё идемпотентно — повторный запуск не плодит дубли и работает как resume)

### Telegram-чат (JSON-экспорт Telegram Desktop, класть в data/tg_chats/)

```powershell
.\.venv\Scripts\python.exe -m src.index add "data\tg_chats\ChatExport.json" --collection scientism --min-chars 20
```

- `--min-chars N` — эмбеддятся сообщения длиной ≥N (единый дефолт 20 для ТГ и YouTube, задан `DEFAULT_MIN_CHARS` в config.py); короткие всё равно попадают в SQLite и видны в `--context`.
- Длинные сообщения (> `MAX_CHUNK_CHARS`=800 символов) дробятся на несколько чанков по границам предложений — так же, как YouTube-комментарии (одиночные чанки сохраняют прежний id, дублей при переиндексации нет). В SQLite сообщение лежит целиком, `--context` показывает его полностью.
- `--link-template` — шаблон ссылки, дефолт `https://t.me/c/{chat_id}/{message_id}` (для приватных супергрупп; chat_id в экспорте уже без префикса -100). Для чатов с публичным username: `--username имя` → ссылки `t.me/имя/<id>`.
- Ориентир скорости: ~230k сообщений ≈ 27 мин на RTX 3070 Ti.

### Комментарии YouTube-видео

```powershell
.\.venv\Scripts\python.exe -m src.index add-yt-comments "https://youtu.be/KjhI4vDfMdY" --collection scientism
```

- Принимает URL (youtu.be, youtube.com/watch, youtube.com/live — мусорные ?si= отбрасываются) или голый video_id.
- Выгрузка через yt-dlp без API-ключа; чанк = один комментарий, но длинные (> `MAX_CHUNK_CHARS`=800 символов) дробятся на несколько чанков по границам предложений — иначе e5 обрезает хвост после ~512 токенов и пухнет VRAM. Ссылки вида `https://www.youtube.com/watch?v=<id>&lc=<comment_id>` открывают конкретный комментарий под видео (у всех частей одного коммента ссылка/`comment_id` общие).
- Если комментарии отключены — пропуск с сообщением, не ошибка.

### Транскрипция YouTube-видео

```powershell
.\.venv\Scripts\python.exe -m src.index add-yt-transcript "https://youtu.be/KjhI4vDfMdY" --collection scientism --lang ru
```

- youtube-transcript-api (ручные и автосубтитры; фолбэк en), запасной путь — yt-dlp vtt.
- Реплики склеиваются в чанки ~500–700 символов, граница по концу предложения или паузе >2с; хранятся start_sec/end_sec.
- Ссылки `https://youtu.be/<id>?t=<сек>` открывают видео с нужного таймкода.
- У стримов транскрипции может не быть — пропуск с сообщением.

### Пакетно: список видео из файла (комменты + транскрипции)

```powershell
.\.venv\Scripts\python.exe scripts\ingest_videos.py docs\video_for_seach_scientism.md --collection scientism
```

Файл — ссылки по одной в строке (`#` — комментарий). Пишет сводную таблицу и `logs/ingest_report.json`. Лог удобно перенаправлять в `logs/ingest_log.txt`.

Работает в две фазы: **(1)** параллельно (по умолчанию 4 потока, `--download-workers N`) скачивает инфо/комменты/субтитры всех видео в кэш `data/youtube/`; **(2)** одним проходом собирает чанки со всех видео и эмбеддит их полными батчами `EMBED_BATCH_SIZE` — маленькие видео больше не гоняют GPU мелкими порциями. Обе фазы идемпотентны (кэш и `existing_ids` пропускают уже сделанное), `store.save` раз в 10 батчей — обрыв не теряет прогресс. Порог коротких — `--min-chars` (дефолт 20).

ВАЖНО: не запускай два процесса индексации одновременно — запись index.faiss неатомарная, параллельные запуски портят индекс.

## Поиск

```powershell
.\.venv\Scripts\python.exe -m src.search "наука единственный способ познания" -k 10 --collection scientism --context 3
```

Опции:
- `-k N` — сколько результатов (дефолт 5).
- `--context M` — M соседей до/после: для tg — соседние сообщения чата; для yt_comments — родительский комментарий + ответы треда; для yt_transcript — соседние фрагменты по времени. Всё из локального SQLite, без сети.
- `--json` — машиночитаемый вывод (с вложенным context).

Фильтры по метаданным (применяются пост-фактум к docstore, БЕЗ пересчёта эмбеддингов и без переиндексации; все комбинируются по И; при любом фильтре поиск углубляет выборку итеративно, поэтому чуть медленнее):
- `--source tg|yt_comments|yt_transcript` — только один тип источника.
- `--author "Имя"` — точное совпадение имени автора (для YouTube это @handle).
- `--chat X` — только этот ТГ-чат: точный `chat_id` ИЛИ подстрока названия чата (регистронезависимо).
- `--video X` — только это YouTube-видео: точный `video_id` ИЛИ подстрока названия видео.
- `--channel X` — только этот YouTube-канал: подстрока названия канала.
- `--filter "поле=значение"` / `--filter "поле~подстрока"` — универсальный фильтр по ЛЮБОМУ полю метаданных (`=` точное совпадение, `~` вхождение подстроки, оба регистронезависимы). Можно повторять — условия складываются по И. Список полей — см. п.4 «Правила для агента». Пример: `--filter "lang=ru" --filter "video_title~Панчин"`.

Откуда брать значения для `--chat`/`--video`/`--channel`: готовый список всех источников коллекции с их id и названиями лежит в `docs/sources_<коллекция>.md`. Обновить его (читает только docstore, без GPU и сети): `.\.venv\Scripts\python.exe scripts\list_sources.py --collection scientism`.

Формат выдачи (каждая карточка начинается с типа источника):

```
#1  score=0.8972  [tg]
Чат: Science Inquisition Comments
Автор: Имя   Дата: 2022-01-04T04:51:37
Текст: ...
Ссылка: https://t.me/c/1545461926/3368

#2  score=0.8858  [yt_comments]
Видео: Александр Панчин — Сциентизм: наука — это религия?
Автор: @handle   Дата: 2025-07-16T00:00:00   Лайков: 2
Текст: ...
Ссылка: https://www.youtube.com/watch?v=KjhI4vDfMdY&lc=Ugy...

#3  score=0.8406  [yt_transcript]
Видео: Александр Панчин — Сциентизм: наука — это религия?
Фрагмент: [52:04–52:46]
Текст: ...
Ссылка: https://youtu.be/KjhI4vDfMdY?t=3124
```

## Правила для агента

1. Всегда указывай `--collection` явно.
2. Кандидата на цитирование НИКОГДА не бери по одному сообщению — читай `--context 3`–`5` и смотри тред: в чатах (особенно Kortik Chat) много сарказма, иронии и мемов; сообщение без контекста регулярно означает противоположное.
3. Формулируй несколько вариаций запроса (5–10): эмбеддинговый поиск чувствителен к формулировке.
4. Метаданные чанков: tg — author, author_id, date, message_id, chat_id, chat_name, reply_to_message_id, link; yt_comments — author, date, likes, video_id, video_title, channel, comment_id, parent_id, link; yt_transcript — video_id, video_title, channel, lang, start_sec, end_sec, link. Поле `source` есть у всех.
5. Дата у YouTube-комментариев приблизительная (yt-dlp отдаёт «N лет назад», округляется до дня).

## Частые проблемы

- `ModuleNotFoundError` → запущен глобальный python; используй `.\.venv\Scripts\python.exe`.
- Кракозябры в консоли → выводи через `-m src.search` (stdout переключается в utf-8 сам); в PowerShell при необходимости `[Console]::OutputEncoding = [Text.Encoding]::UTF8`.
- Поиск с `--source yt_transcript` медленнее обычного: фильтрация по редкому источнику углубляет выборку — это нормально.
- Прервали индексацию — просто повтори команду: уже добавленное пропустится (id чанков детерминированные).
- yt-dlp начал падать на всех видео → YouTube поменял API, обнови: `.\.venv\Scripts\python.exe -m pip install -U yt-dlp`.
- Предупреждение «Support for Python version 3.10 has been deprecated» от yt-dlp — безвредно.
- `faiss.read_index` падает с `size >= 0 && size < (1<<40)` → index.faiss повреждён (обычно после параллельных запусков индексации). Лечение: удалить `store/<имя>/index.faiss` И `docstore.pkl` (raw.sqlite и кэш data/ можно оставить), затем переиндексировать все источники заново.
