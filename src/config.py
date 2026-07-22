from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Корень проекта (папка, содержащая src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Личные настройки пользователя (не в git) — см. .env.example.
# override=False: реальные переменные окружения выигрывают у .env.
load_dotenv(PROJECT_ROOT / ".env", override=False)

# Где лежат все коллекции (индексы + сырые сообщения).
STORE_ROOT = PROJECT_ROOT / "store"

# Первоисточники: экспорты Telegram и кэш YouTube (по подпапке на тип источника).
DATA_ROOT = PROJECT_ROOT / "data"
TG_CHATS_ROOT = DATA_ROOT / "tg_chats"
YOUTUBE_ROOT = DATA_ROOT / "youtube"

# Логи и отчёты индексации.
LOGS_ROOT = PROJECT_ROOT / "logs"

# e5-модели требуют префиксы "query:" / "passage:" для корректной работы.
EMBEDDING_QUERY_PREFIX = "query: "
EMBEDDING_PASSAGE_PREFIX = "passage: "

# Сколько сообщений накапливать перед вызовом encode (length-sorting внутри ST
# делает большие порции эффективнее; сам GPU-батч задаётся GPU_BATCH_SIZE ниже).
EMBED_BATCH_SIZE = 1024

# Внутренний батч GPU (передаётся в model.encode). ВАЖНО: не увеличивать бездумно —
# на 8GB VRAM батч из длинных текстов (до 512 токенов e5, напр. «простыни»-комментарии
# YouTube) переполняет память и включает CUDA sysmem fallback (спилл в RAM через PCIe)
# => замедление в ~50 раз, процесс выглядит зависшим. 16 — безопасно для 8GB, скорость
# почти не падает. Короткие ТГ-сообщения потолок не задевают, для них можно и больше.
GPU_BATCH_SIZE = 16

# Минимум символов текста, чтобы чанк попал в эмбеддинги (короче — только в SQLite
# для контекста). Единый дефолт для ТГ-сообщений и YouTube-комментариев.
DEFAULT_MIN_CHARS = 20

# Максимальная длина одного чанка в символах (для всех источников: ТГ-сообщения и
# YouTube-комментарии). Текст длиннее режется на несколько чанков по границам предложений:
# иначе e5 обрезает всё после ~512 токенов (теряется смысл конца) и раздувается пиковая VRAM
# на батче «простыней». ~800 символов кириллицы ≈ под 512 токенов с запасом.
MAX_CHUNK_CHARS = 800

# Дефолтный шаблон ссылки на сообщение.
DEFAULT_LINK_TEMPLATE = "https://t.me/c/{chat_id}/{message_id}"

DEFAULT_COLLECTION = os.environ.get("RAG_DEFAULT_COLLECTION", "default")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
