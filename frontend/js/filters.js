// Наполнение управляющих элементов (коллекции, чаты, каналы, видео) данными
// сервера и сбор текущих значений фильтров в объект запроса для /api/search.

import { fetchCollections, fetchSources } from "./api.js";

const el = (id) => document.getElementById(id);

function fillSelect(select, items, { value, label, keepFirst = true } = {}) {
  const first = keepFirst ? select.firstElementChild : null;
  select.innerHTML = "";
  if (first) select.appendChild(first);
  for (const it of items) {
    const opt = document.createElement("option");
    opt.value = value(it);
    opt.textContent = label(it);
    select.appendChild(opt);
  }
}

export async function loadCollections() {
  const collections = await fetchCollections();
  const sel = el("collection");
  sel.innerHTML = "";
  for (const c of collections) {
    const opt = document.createElement("option");
    opt.value = c.name;
    opt.textContent =
      c.vectors != null ? `${c.name} (${c.vectors.toLocaleString("ru")})` : c.name;
    sel.appendChild(opt);
  }
  return collections;
}

// Подтягивает чаты/каналы/видео выбранной коллекции в выпадающие фильтры.
export async function loadSources(collection) {
  const data = await fetchSources(collection);

  fillSelect(el("chat"), data.tg_chats || [], {
    value: (c) => c.chat_id,
    label: (c) => `${c.chat_name || c.chat_id} (${c.count})`,
  });
  fillSelect(el("channel"), data.channels || [], {
    value: (c) => c.channel,
    label: (c) => `${c.channel} (${c.count})`,
  });
  fillSelect(el("video"), data.videos || [], {
    value: (v) => v.video_id,
    label: (v) => {
      const t = v.video_title || v.video_id;
      return t.length > 60 ? t.slice(0, 57) + "…" : t;
    },
  });
  return data;
}

// Собирает тело запроса из полей формы. Пустые значения опускаются.
export function collectParams() {
  const params = {
    query: el("query").value,
    collection: el("collection").value,
    k: Number(el("k").value) || 5,
    context: Number(el("context").value) || 0,
  };
  const source = el("source").value;
  const chat = el("chat").value;
  const channel = el("channel").value;
  const video = el("video").value;
  const author = el("author").value.trim();
  if (source) params.source = source;
  if (chat) params.chat = chat;
  if (channel) params.channel = channel;
  if (video) params.video = video;
  if (author) params.author = author;
  return params;
}
