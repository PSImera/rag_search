// Точка входа: инициализация фильтров, обработка сабмита, индикатор статуса.

import { search } from "./api.js";
import {
  loadCollections,
  loadSources,
  collectParams,
  filterVideosByChannel,
} from "./filters.js";
import { renderResults, renderPlaceholder } from "./results.js";

const el = (id) => document.getElementById(id);
let searching = false;

function setStatus(text, isError = false) {
  const s = el("status");
  if (!text) {
    s.hidden = true;
    return;
  }
  s.hidden = false;
  s.textContent = text;
  s.classList.toggle("status--error", isError);
}

async function refreshSources() {
  const collection = el("collection").value;
  if (!collection) return;
  try {
    await loadSources(collection);
  } catch (e) {
    setStatus(`Не удалось загрузить фильтры: ${e.message}`, true);
  }
}

async function runSearch(ev) {
  if (ev) ev.preventDefault();
  if (searching) return;
  const params = collectParams();
  if (!params.query.trim()) {
    el("query").focus();
    return;
  }
  searching = true;
  el("search-btn").disabled = true;
  setStatus("Поиск…");
  try {
    const data = await search(params);
    renderResults(data.results || []);
    setStatus(`Найдено: ${data.count}`);
    setTimeout(() => setStatus(""), 2500);
  } catch (e) {
    renderPlaceholder("Ошибка запроса.");
    setStatus(`Ошибка: ${e.message}`, true);
  } finally {
    searching = false;
    el("search-btn").disabled = false;
  }
}

async function init() {
  el("search-form").addEventListener("submit", runSearch);
  el("collection").addEventListener("change", refreshSources);
  el("channel").addEventListener("change", (e) =>
    filterVideosByChannel(e.target.value)
  );

  try {
    const collections = await loadCollections();
    if (!collections.length) {
      renderPlaceholder("Нет ни одной коллекции в store/.");
      return;
    }
    await refreshSources();
    renderPlaceholder("Введите запрос и нажмите «Искать».");
  } catch (e) {
    setStatus(`Не удалось получить список коллекций: ${e.message}`, true);
    renderPlaceholder("Сервер недоступен.");
  }
}

init();
