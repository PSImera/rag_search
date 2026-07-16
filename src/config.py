from __future__ import annotations

from pathlib import Path

# Корень проекта (папка, содержащая src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Где лежат все коллекции (индексы + сырые сообщения).
STORE_ROOT = PROJECT_ROOT / "store"

# Первоисточники: экспорты Telegram и кэш YouTube (по подпапке на тип источника).
DATA_ROOT = PROJECT_ROOT / "data"
TG_CHATS_ROOT = DATA_ROOT / "tg_chats"
YOUTUBE_ROOT = DATA_ROOT / "youtube"

# Логи и отчёты индексации.
LOGS_ROOT = PROJECT_ROOT / "logs"

# Локальная модель эмбеддингов (уже закеширована в ~/.cache/huggingface).
# multilingual-e5-large — сильная многоязычная модель, отлично держит русский.
# Работает офлайн, токены платного API не тратятся.
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# e5-модели требуют префиксы "query:" / "passage:" для корректной работы.
EMBEDDING_QUERY_PREFIX = "query: "
EMBEDDING_PASSAGE_PREFIX = "passage: "

# Сколько сообщений накапливать перед вызовом encode (length-sorting внутри ST
# делает большие порции эффективнее; GPU-батч задаётся отдельно в embeddings.py).
EMBED_BATCH_SIZE = 1024

# Дефолтная коллекция.
DEFAULT_COLLECTION = "default"

# Дефолтный шаблон ссылки на сообщение.
DEFAULT_LINK_TEMPLATE = "https://t.me/c/{chat_id}/{message_id}"
