// Tiny DOM and formatting helpers shared by both apps. No framework: views return nodes.

export function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v; // only for trusted inline SVG icons
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  append(el, kids);
  return el;
}

function append(el, kids) {
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
}

export function mount(target, ...nodes) {
  target.replaceChildren();
  append(target, nodes);
}

const ICONS = {
  today: '<path d="M4 6h16M4 12h16M4 18h10"/>',
  teach: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3" fill="currentColor"/>',
  send: '<path d="M12 19V5M5 12l7-7 7 7"/>',
  accounts: '<rect x="5" y="3" width="14" height="18" rx="3"/><path d="M10 18h4"/>',
  chev: '<path d="M9 6l6 6-6 6"/>',
  back: '<path d="M15 6l-6 6 6 6"/>',
  check: '<path d="M5 12l5 5 9-10"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  video: '<rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10l5-3v10l-5-3"/>',
  chat: '<path d="M4 5h16v11H9l-5 4z"/>',
  progress: '<path d="M4 19l5-6 4 3 7-9"/>',
  lock: '<rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>',
};

export function icon(name, size = 20) {
  return h("span", {
    "aria-hidden": "true",
    style: { display: "inline-flex", flex: "none" },
    html: `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[name] || ""}</svg>`,
  });
}

let toastTimer;
export function toast(text, tone = "") {
  let wrap = document.querySelector(".toast-wrap");
  if (!wrap) {
    wrap = h("div", { class: "toast-wrap", role: "status", "aria-live": "polite" });
    document.body.append(wrap);
  }
  mount(wrap, h("div", { class: `toast ${tone}` }, text));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => wrap.replaceChildren(), 4200);
}

// Run an async action from a button: busy state, error toast, no double submit.
export async function busy(button, fn) {
  if (button?.getAttribute("aria-busy") === "true") return;
  button?.setAttribute("aria-busy", "true");
  try {
    return await fn();
  } catch (err) {
    toast(err.message || "Something went wrong. Try again.", "bad");
  } finally {
    button?.removeAttribute("aria-busy");
  }
}

export function when(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(+d)) return "";
  const diff = (Date.now() - d) / 1000;
  const abs = Math.abs(diff);
  const rel = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  if (abs < 60) return diff >= 0 ? "just now" : "in a moment";
  if (abs < 3600) return rel.format(-Math.round(diff / 60), "minute");
  if (abs < 86400) return rel.format(-Math.round(diff / 3600), "hour");
  if (abs < 86400 * 6) return rel.format(-Math.round(diff / 86400), "day");
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function clock(iso) {
  if (!iso) return "";
  return new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

export function bytes(n) {
  if (!n && n !== 0) return "";
  if (n < 1024 * 1024) return `${Math.max(1, Math.round(n / 1024))} KB`;
  return `${(n / 1024 / 1024).toFixed(n < 10 * 1024 * 1024 ? 1 : 0)} MB`;
}

export function plural(n, one, many = one + "s") {
  return `${n} ${n === 1 ? one : many}`;
}

export function hashRoute() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [path, qs] = raw.split("?");
  return { parts: path.split("/").filter(Boolean), query: new URLSearchParams(qs || "") };
}

export function go(path) {
  location.hash = path.startsWith("#") ? path : `#${path}`;
}

// A form is "dirty" once the person types; background refresh must not wipe their input.
export function editing() {
  const a = document.activeElement;
  return !!(a?.matches?.("input, textarea, select") || document.querySelector("[data-dirty]"));
}
