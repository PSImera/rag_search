// Рендер карточек выдачи. Три типа источника (tg / yt_comments / yt_transcript)
// оформляются по-разному; контекст (соседи/тред) сворачивается в <details>.

const SRC_LABEL = {
  tg: "Telegram",
  yt_comments: "YT-комментарий",
  yt_transcript: "YT-транскрипт",
};

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

function mmss(sec) {
  sec = Math.floor(Number(sec) || 0);
  const s = sec % 60;
  const m = Math.floor(sec / 60) % 60;
  const h = Math.floor(sec / 3600);
  const pad = (n) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

function metaRow(parts) {
  const cells = parts
    .filter((p) => p != null && p !== "")
    .map((p) => `<span>${esc(p)}</span>`)
    .join("");
  return `<div class="card__meta">${cells}</div>`;
}

function linkTag(href) {
  if (!href) return "";
  return `<a class="card__link" href="${esc(href)}" target="_blank" rel="noopener">${esc(
    href
  )}</a>`;
}

function ctxItem(text, strong) {
  const body = strong ? `<b>${esc(strong)}</b> ${esc(text)}` : esc(text);
  return `<div class="ctx-item">${body}</div>`;
}

function renderContext(r) {
  const ctx = r.context;
  if (!ctx) return "";
  const src = r.source || "tg";
  let groups = "";

  if (src === "yt_comments") {
    if (ctx.parent) {
      groups += `<div class="ctx-group"><div class="ctx-group__label">родительский</div>${ctxItem(
        ctx.parent.text,
        ctx.parent.author
      )}</div>`;
    }
    if (ctx.replies && ctx.replies.length) {
      const items = ctx.replies.map((c) => ctxItem(c.text, c.author)).join("");
      groups += `<div class="ctx-group"><div class="ctx-group__label">тред</div>${items}</div>`;
    }
  } else if (src === "yt_transcript") {
    for (const [label, key] of [["до", "before"], ["после", "after"]]) {
      const items = ctx[key] || [];
      if (!items.length) continue;
      const rows = items
        .map((c) => ctxItem(c.text, `[${mmss(c.start_sec)}]`))
        .join("");
      groups += `<div class="ctx-group"><div class="ctx-group__label">${label}</div>${rows}</div>`;
    }
  } else {
    for (const [label, key] of [["до", "before"], ["после", "after"]]) {
      const items = ctx[key] || [];
      if (!items.length) continue;
      const rows = items.map((m) => ctxItem(m.text, m.author)).join("");
      groups += `<div class="ctx-group"><div class="ctx-group__label">${label}</div>${rows}</div>`;
    }
  }

  if (!groups) return "";
  return `<details class="context"><summary>контекст</summary>${groups}</details>`;
}

function renderCard(r, i) {
  const src = r.source || "tg";
  const head = `<div class="card__head">
      <span class="badge badge--${src}">${SRC_LABEL[src] || src}</span>
      <span class="card__rank">#${i}</span>
      <span class="card__score">score ${Number(r.score).toFixed(4)}</span>
    </div>`;

  let title = "";
  let meta = "";

  if (src === "yt_transcript") {
    title = `<div class="card__title">${esc(r.channel)} — ${esc(r.video_title)}</div>`;
    meta = metaRow([`Фрагмент [${mmss(r.start_sec)}–${mmss(r.end_sec)}]`, r.lang]);
  } else if (src === "yt_comments") {
    title = `<div class="card__title">${esc(r.channel)} — ${esc(r.video_title)}</div>`;
    meta = metaRow([r.author, r.date, r.likes != null ? `♥ ${r.likes}` : null]);
  } else {
    title = `<div class="card__title">${esc(r.chat_name || "Telegram")}</div>`;
    meta = metaRow([r.author, r.date]);
  }

  return `<article class="card card--${src}">
    ${head}
    ${title}
    ${meta}
    <div class="card__text">${esc(r.text)}</div>
    ${linkTag(r.link)}
    ${renderContext(r)}
  </article>`;
}

export function renderResults(results) {
  const root = document.getElementById("results");
  if (!results.length) {
    root.innerHTML = `<div class="placeholder">Ничего не найдено. Попробуйте переформулировать запрос или ослабить фильтры.</div>`;
    return;
  }
  root.innerHTML = results.map((r, idx) => renderCard(r, idx + 1)).join("");
}

export function renderPlaceholder(text) {
  document.getElementById("results").innerHTML = `<div class="placeholder">${esc(
    text
  )}</div>`;
}
