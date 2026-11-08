// The only place the UI talks to the server. `?example` in the URL swaps the server for an
// in-browser fixture (assets/example-*.js): nothing is sent, nothing is stored, every screen says so.

export const EXAMPLE = new URLSearchParams(location.search).has("example");

export class ApiError extends Error {
  constructor(status, message, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

let fixture = null; // (method, path, body) => result, set by the example module

export function useFixture(handler) {
  fixture = handler;
}

export async function api(method, path, body) {
  if (EXAMPLE) {
    if (!fixture) throw new ApiError(500, "Example data isn't loaded.");
    await new Promise((r) => setTimeout(r, 120));
    return structuredClone(await fixture(method, path, body === undefined ? undefined : structuredClone(body)));
  }
  let res;
  try {
    res = await fetch(path, {
      method,
      credentials: "same-origin",
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "Can't reach the workspace. Check your connection and try again.");
  }
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (!res.ok) throw new ApiError(res.status, data?.error || friendly(res.status), data);
  return data;
}

function friendly(status) {
  if (status === 401) return "Please sign in again with your link.";
  if (status === 403) return "You don't have access to that.";
  if (status === 404) return "That isn't available on this server yet.";
  if (status === 409) return "This changed since you opened it. Review the latest version.";
  if (status === 413) return "That file is too large.";
  return "Something went wrong on our side. Try again in a minute.";
}

export const get = (p) => api("GET", p);
export const post = (p, b = {}) => api("POST", p, b);
export const patch = (p, b) => api("PATCH", p, b);
export const del = (p) => api("DELETE", p);

// Uploads are raw bodies (INTERFACES.md: PUT /api/media?name=…). XHR for progress events.
export function upload(file, onProgress = () => {}) {
  if (EXAMPLE) {
    onProgress(1);
    return Promise.resolve({
      path: `example/${file.name}`,
      sha256: fakeSha(`${file.name}:${file.size}:${file.lastModified}`),
      name: file.name,
      bytes: file.size,
      content_type: file.type,
      local_url: URL.createObjectURL(file), // stays in this browser tab
    });
  }
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", `/api/media?name=${encodeURIComponent(file.name)}`);
    xhr.withCredentials = true;
    if (file.type) xhr.setRequestHeader("Content-Type", file.type);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onerror = () => reject(new ApiError(0, "Upload interrupted. Check your connection and try again."));
    xhr.onload = () => {
      let data = null;
      try {
        data = JSON.parse(xhr.responseText || "null");
      } catch {}
      if (xhr.status >= 200 && xhr.status < 300) resolve({ ...data, local_url: URL.createObjectURL(file) });
      else reject(new ApiError(xhr.status, data?.error || friendly(xhr.status), data));
    };
    xhr.send(file);
  });
}

// Where to show a stored media item from. Example items have no server copy.
export function mediaSrc(m) {
  if (!m) return null;
  if (m.local_url) return m.local_url;
  if (EXAMPLE) return null;
  if (m.url) return m.url;
  return m.sha256 ? `/api/media/${m.sha256}` : null;
}

// Where a visitor came from (?src=… on the link we shared) and who referred them (?ref=<person id>).
export const ATTRIBUTION = (() => {
  const q = new URLSearchParams(location.search);
  return { source: q.get("src") || q.get("source") || null, ref: q.get("ref") || null };
})();

// One landing view for the funnel (growth.py stores only a keyed hash of this random id; no IP, no URL).
// Never from the example, and a missing growth lane is not an error.
export function visit(page) {
  if (EXAMPLE) return;
  let id = null;
  try {
    id = localStorage.getItem("tw_visitor") || crypto.randomUUID();
    localStorage.setItem("tw_visitor", id);
  } catch {
    id = `v${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
  }
  api("POST", "/api/growth/visit", { visitor: id, page, source: ATTRIBUTION.source }).catch(() => {});
}

// Keep ?example on every in-app link.
export function href(path) {
  return `${EXAMPLE ? "?example" : ""}#${path.replace(/^#/, "")}`;
}

function fakeSha(s) {
  let out = "";
  for (let seed = 0; out.length < 64; seed++) {
    let x = 2166136261 ^ seed;
    for (let i = 0; i < s.length; i++) x = Math.imul(x ^ s.charCodeAt(i), 16777619);
    out += (x >>> 0).toString(16).padStart(8, "0");
  }
  return out.slice(0, 64);
}
export { fakeSha };
