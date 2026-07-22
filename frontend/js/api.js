// Обёртки над JSON-API сервера (src/server.py). Ничего не рендерят —
// только сеть и разбор ответа, ошибки пробрасываются наверх.

async function asJson(resp) {
  let data = null;
  try {
    data = await resp.json();
  } catch (_) {
    /* тело не JSON */
  }
  if (!resp.ok) {
    const msg = (data && data.error) || `HTTP ${resp.status}`;
    throw new Error(msg);
  }
  return data;
}

export async function fetchCollections() {
  const data = await asJson(await fetch("/api/collections"));
  return data.collections || [];
}

export async function fetchSources(collection) {
  const url = `/api/sources?collection=${encodeURIComponent(collection)}`;
  return asJson(await fetch(url));
}

export async function search(params) {
  const resp = await fetch("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return asJson(await resp);
}
