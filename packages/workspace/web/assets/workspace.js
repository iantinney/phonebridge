// Phonebridge workspace: see the real phone, start a supported task, follow each step with the frame it saw, stop it
// or take the phone back. Contract: INTERFACES.md §Workspace (tw-integrator's workspace.py). Nothing here is
// simulated: a route, device or task kind that isn't there says so, with the reason the server gives.
import { get, post, del as api_delete, ApiError } from "./api.js";
import { h, mount, toast, busy, when } from "./ui.js";
import { initTeach, openTeach, teachStateChanged } from "./workspace-teach.js";

const side = document.getElementById("side");
const stage = document.getElementById("stage");
const panel = document.getElementById("panel");

const S = {
  devices: null, problem: null, deviceId: null,
  kinds: null, kindsKey: "", kindsProblem: null, kind: null, draft: {},
  tasks: [], taskId: null, task: null,
  view: null, // a step's stored frame, shown instead of the current one
  mode: "run", // the right panel: run a task, or teach a workflow
  releasing: {}, // task id → "stop" | "handoff" while the request waits for the task to actually end
  names: {}, // routine id → its name, for task titles
  stepsOpen: {}, // task id → the person opened (true) or closed (false) its step list
  live: { url: null, ageAtRx: null, rxAt: 0, capturedAt: null, source: null, err: null, times: [] },
};
const visible = () => document.visibilityState === "visible";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const dev = () => S.devices?.find((d) => d.id === S.deviceId) || null;
// The server withholds the screen while the phone is in personal use (no task holds it); then nothing is polled.
const shared = (d) => d && d.connection?.state === "connected" && d.frame?.available !== false;
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
const OPEN = ["queued", "running", "needs"];

const CONN = {
  connected: ["Connected", "ok"], locked: ["Locked", "warn"], unplugged: ["Unplugged", "bad"],
  untrusted: ["Not trusted", "warn"], asleep: ["Mac asleep", "bad"], unknown: ["Unknown", "warn"],
};
const KIND = {
  observe: { label: "Read the screen", blurb: "One look at the phone: the frame and the text on it, kept as evidence." },
  account_check: { label: "Account check", blurb: "Open named Instagram profiles and report what is visibly there.",
    examples: [["Check @nasa and @natgeo", { handles: "nasa, natgeo" }]] },
  feed_scout: { label: "Feed scout", blurb: "Browse a bounded stretch of the feed for posts that match your request.",
    examples: [["Home espresso setups, 15 posts", { request: "home espresso setups", limit: "15" }]] },
  taught_run: { label: "Run a saved workflow", blurb: "Pick one of your saved workflows and give it fresh inputs." },
  coach_observe: { label: "Coach this conversation", blurb: "Open a conversation on the phone first. It reads what's on screen, then you can ask about it here; every answer points to the message it rests on." },
};
const APPS = {
  "com.apple.springboard": "Home Screen", "com.burbn.instagram": "Instagram", "com.zhiliaoapp.musically": "TikTok",
  "com.atebits.Tweetie2": "X", "co.hinge.app": "Hinge", "com.cardify.tinder": "Tinder", "com.bumble.app": "Bumble",
  "com.apple.MobileSMS": "Messages", "com.apple.mobilesafari": "Safari", "com.google.Gmail": "Gmail", "com.apple.mobilemail": "Mail",
  "com.apple.Preferences": "Settings", "com.apple.camera": "Camera", "com.apple.mobileslideshow": "Photos",
  "com.google.ios.youtube": "YouTube", "net.whatsapp.WhatsApp": "WhatsApp", "com.facebook.Facebook": "Facebook",
};
const appName = (id) => (!id ? null : APPS[id] || (/^[\w-]+(\.[\w-]+){2,}$/.test(id) ? id.split(".").pop().replace(/^./, (c) => c.toUpperCase()) : id));
const kindLabel = (k) => KIND[k]?.label || k.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
const STATE = { queued: "Queued", running: "Working", needs: "Needs you", done: "Done", failed: "Failed", stopped: "Stopped", handed_off: "Handed off" };

const ICON = {
  observe: '<circle cx="12" cy="12" r="3"/><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/>',
  tap: '<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="8"/>',
  swipe: '<path d="M5 12h14M15 8l4 4-4 4"/>',
  type: '<rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 10h.01M11 10h.01M15 10h.01M8 14h8"/>',
  open_app: '<rect x="4" y="4" width="16" height="16" rx="4"/><path d="M10 14l4-4M10 10h4v4"/>',
  wait: '<circle cx="12" cy="12" r="8"/><path d="M12 8v4l3 2"/>',
  note: '<path d="M5 5h14v10H9l-4 4z"/>',
  message: '<path d="M5 5h14v10H9l-4 4z"/>',
  result: '<path d="M5 12l5 5 9-10"/>',
};
const svg = (name, size = 14) => h("span", { class: "ic", "aria-hidden": "true",
  html: `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICON[name] || ICON.note}</svg>` });
const hhmmss = (iso) => (iso ? new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23" }) : "");
const secs = (s) => (s == null ? "" : s < 10 ? `${s.toFixed(1)} s` : s < 90 ? `${Math.round(s)} s` : when(new Date(Date.now() - s * 1000).toISOString()));
const taskTitle = (t) => {
  const i = t.input || {};
  if (i.routine_id) {
    nameRoutine(i.routine_id);
    const p = i.params || i.inputs || {};
    const arg = p.handle ? `@${String(p.handle).replace(/^@/, "")}` : Object.values(p).filter((x) => typeof x === "string")[0] || "";
    return [S.names[i.routine_id] || "Saved workflow", arg].filter(Boolean).join(" · ");
  }
  const what = i.handles ? [].concat(i.handles).map((x) => `@${String(x).replace(/^@/, "")}`).join(", ") : i.request || i.routine_name || "";
  return what ? `${kindLabel(t.kind)} · ${what}` : kindLabel(t.kind);
};

// ---------- sidebar ----------

function renderSide() {
  const devices = S.devices || [];
  mount(side,
    h("div", { class: "brand" }, h("span", { class: "brand-mark", "aria-hidden": "true" }), "Phonebridge"),
    h("div", { class: "side-section" },
      h("div", { class: "label" }, "Phones"),
      S.problem ? h("p", { class: "empty-note" }, S.problem.short)
        : !S.devices ? h("p", { class: "empty-note" }, "Looking for phones…")
        : !devices.length ? h("p", { class: "empty-note" }, "No phone is attached to this computer's bridge.")
        : devices.map(deviceRow)),
    h("div", { class: "side-section" },
      h("div", { class: "label" }, h("span", {}, "Tasks"), S.tasks.length ? h("span", { class: "num" }, S.tasks.length) : null),
      taskRows()),
    S.extras?.length ? h("div", { class: "side-foot" }, S.extras.map(([href, label]) => h("a", { href }, label))) : null);
}

function deviceRow(d) {
  const [label, tone] = CONN[d.connection?.state] || CONN.unknown;
  const beat = d.connection?.heartbeat_age_s;
  return h("button", { class: "row-btn", "aria-current": String(d.id === S.deviceId), onclick: () => selectDevice(d.id) },
    h("span", { class: `dot ${d.connection?.state === "connected" ? "connected" : tone}` }),
    h("span", { class: "grow" },
      h("div", { class: "t" }, d.model || d.name || d.id),
      h("div", { class: "s" }, d.task_id ? "Working on a task" : `${label}${beat != null ? ` · ${secs(beat)}` : ""}`)));
}

function taskRows() {
  if (!S.deviceId) return h("p", { class: "empty-note" }, "Pick a phone to see its tasks.");
  if (!S.tasks.length) return h("p", { class: "empty-note" }, "Nothing has run on this phone from here yet.");
  return S.tasks.slice(0, 40).map((t) => h("button", { class: "row-btn", "aria-current": String(t.id === S.taskId), onclick: () => selectTask(t.id) },
    h("span", { class: "grow" },
      h("div", { class: "t" }, taskTitle(t)),
      h("div", { class: "s" }, h("span", { class: `state ${t.state}` }, STATE[t.state] || t.state), " · ", when(t.created)))));
}

// ---------- stage: the real phone ----------

const frameImg = h("img", { alt: "The phone's screen" });
const tapmark = h("span", { class: "tapmark", hidden: true });
const stamp = h("span", { class: "stamp num", hidden: true });
const overlay = h("div", { class: "overlay", hidden: true });
const screen = h("div", { class: "screen" }, frameImg, tapmark, overlay, stamp);
const device = h("div", { class: "device" }, h("div", { class: "bezel" }, screen));
const stageHead = h("header", { class: "stage-head" });
const fresh = h("div", { class: "fresh num", "aria-live": "off" });
const stageFoot = h("div", { class: "stage-foot" });
mount(stage, stageHead, h("div", { class: "stage-body" }, device), h("div", { class: "stage-foot", style: { paddingBottom: 0 } }, fresh), stageFoot);

function renderStageHead() {
  const d = dev();
  const picker = (S.devices || []).length > 1 ? h("select", { class: "picker", "aria-label": "Phone", onchange: (e) => selectDevice(e.target.value) },
    S.devices.map((x) => h("option", { value: x.id, selected: x.id === S.deviceId }, x.model || x.name || x.id))) : null;
  if (!d) {
    mount(stageHead, h("h1", {}, "Phonebridge"), h("span", { class: "spacer" }),
      S.problem ? h("span", { class: "chip bad" }, S.problem.short) : null);
    return;
  }
  const c = d.connection || {};
  const [label, tone] = CONN[c.state] || CONN.unknown;
  const facts = [
    ["Status", c.reason || label],
    ["Mac checked in", c.heartbeat_age_s != null ? `${secs(c.heartbeat_age_s)} ago` : "unknown"],
    ["Control", d.lease ? (d.lease.mine ? "you" : d.lease.holder ? `held by ${d.lease.holder}` : "free") : "not reported"],
    ["Focus", d.focus ? (d.focus.open ? `work session until ${hhmmss(d.focus.until)}` : "no work session") : "not reported"],
    ["Phone", [d.model, d.ios ? `iOS ${d.ios}` : null, d.name && d.name !== d.model ? `named “${d.name}”` : null].filter(Boolean).join(" · ")],
    ["Open app", d.foreground_app ? `${appName(d.foreground_app)} (${d.foreground_app})` : "unknown"],
  ];
  mount(stageHead,
    h("h1", {}, d.model || d.name || d.id),
    d.foreground_app ? h("span", { class: "app", title: d.foreground_app }, appName(d.foreground_app)) : null,
    picker,
    h("span", { class: "spacer" }),
    h("span", { class: `chip ${tone}`, title: facts.map((f) => f.join(": ")).join("\n") }, label),
    h("button", { class: "btn quiet sm", "aria-expanded": String(!!S.details), onclick: () => { S.details = !S.details; renderStageHead(); } }, "Details"),
    S.details ? h("dl", { class: "kv head-details small" }, facts.map(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])) : null);
}

function renderStageFoot() {
  const d = dev();
  const running = d && S.tasks.find((t) => t.id === d.task_id);
  const canRead = d && !S.problem && d.connection?.state === "connected";
  mount(stageFoot,
    running ? h("span", { class: "small" }, h("span", { class: `state ${running.state}` }, STATE[running.state]), " ", taskTitle(running)) : null,
    ...(running ? stopControls(running) : []),
    d && !running ? h("button", { class: "btn", disabled: !canRead, title: canRead ? "Take one look and read the text on it" : d.connection?.reason || "",
      onclick: (e) => busy(e.currentTarget, observeNow) }, svg("observe"), "Read the screen") : null,
    S.view ? h("button", { class: "btn quiet", onclick: () => { S.view = null; renderFrame(); renderTask(); } }, "Back to the current screen") : null);
}

// What the big screen shows: a step's stored frame when one is picked, else the current frame and its honest age.
function renderFrame() {
  const d = dev();
  const L = S.live;
  tapmark.hidden = true;
  overlay.hidden = true;
  stamp.hidden = true;
  frameImg.style.opacity = "1";
  if (S.view) {
    frameImg.hidden = false;
    frameImg.src = S.view.url;
    stamp.className = "stamp num";
    stamp.textContent = S.view.label || `Step ${S.view.idx + 1} · saw this at ${hhmmss(S.view.at)}`;
    stamp.hidden = false;
    const place = () => {
      const p = S.view?.point || (S.view?.pointPx && frameImg.naturalWidth
        ? { x: S.view.pointPx.x / frameImg.naturalWidth, y: S.view.pointPx.y / frameImg.naturalHeight } : null);
      if (p) Object.assign(tapmark.style, { left: `${p.x * 100}%`, top: `${p.y * 100}%` }), (tapmark.hidden = false);
    };
    if (frameImg.complete && frameImg.naturalWidth) place();
    else frameImg.onload = () => { frameImg.onload = null; if (S.view) place(); };
    fresh.textContent = S.view.caption || "Evidence frame from this task"; // each opener says what this frame is
    return;
  }
  if (L.url && frameImg.src !== L.url) frameImg.src = L.url;
  if (!L.url) frameImg.removeAttribute("src");
  frameImg.hidden = !L.url;
  const problem = S.problem ? [S.problem.title, S.problem.message]
    : !d ? ["No phone selected", ""]
    : d.connection?.state !== "connected" ? [CONN[d.connection?.state]?.[0] || "Not connected", d.connection?.reason || ""]
    : d.frame?.available === false ? ["Screen not shared", `${cap(d.frame.reason || "the server isn't sharing this screen")}. It appears while a task holds the phone.`]
    : !L.url && L.err ? ["No picture from the phone", L.err]
    : !L.url ? ["Connecting to the screen…", ""] : null;
  if (problem) {
    mount(overlay, h("div", {}, h("b", {}, problem[0]), problem[1]));
    overlay.hidden = false;
    if (L.url) frameImg.style.opacity = "0.35";
  }
  const age = L.url ? L.ageAtRx + (performance.now() - L.rxAt) / 1000 : null;
  const fps = L.times.length > 1 ? (L.times.length - 1) / ((L.times.at(-1) - L.times[0]) / 1000) : null;
  if (age != null) {
    const paused = !visible();
    const live = !problem && !paused && age < 3;
    stamp.className = `stamp num ${age >= 15 || problem ? "bad" : "warn"}`;
    stamp.textContent = paused ? "Paused" : age >= 15 ? "Stale" : `${secs(age)} old`;
    stamp.hidden = live;
    mount(fresh, live ? h("span", { class: "live-dot" }) : null,
      live ? (L.pixelKnown ? `Live · ${secs(age)} old` : `Live · updated ${secs(age)} ago`) : paused ? "Paused while this tab is hidden" : `Received ${secs(age)} ago`);
    fresh.title = [`Captured ${hhmmss(L.capturedAt)}`, fps ? `about ${fps.toFixed(1)} frames a second` : null,
      L.source ? `source: ${L.source}` : null, L.err || null].filter(Boolean).join("\n");
  } else (fresh.textContent = ""), (fresh.title = "");
}

// About one frame a second while this tab is visible and the phone is connected; nothing otherwise.
async function frameLoop() {
  for (;;) {
    const d = dev();
    if (!visible() || !d || S.problem || !shared(d)) { await sleep(800); continue; }
    const t0 = performance.now();
    try {
      const ac = new AbortController();
      const timer = setTimeout(() => ac.abort(), 10000);
      const res = await fetch(`/api/devices/${encodeURIComponent(d.id)}/frame`, { cache: "no-store", credentials: "same-origin", signal: ac.signal })
        .catch((e) => { throw new Error(e.name === "AbortError" ? "The phone's screen didn't answer within 10 s." : "Can't reach the server."); })
        .finally(() => clearTimeout(timer));
      if (!res.ok) {
        let msg = `The screen can't be read right now (${res.status}).`;
        try { msg = (await res.json()).error || msg; } catch {}
        throw new Error(res.status === 404 ? "No frame from this phone yet." : msg);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.src = url;
      await img.decode();
      const fetchS = (performance.now() - t0) / 1000;
      const old = S.live.url;
      // pixel age when the bridge knows it; otherwise only how long ago the VM fetched it (said as "updated")
      const pixel = parseFloat(res.headers.get("X-Pixel-Age-S") ?? res.headers.get("X-Age-S"));
      S.live = { ...S.live, url, err: null, rxAt: performance.now(), source: res.headers.get("X-Source"),
        capturedAt: res.headers.get("X-Fetched-At") || res.headers.get("X-Captured-At") || new Date().toISOString(),
        pixelKnown: Number.isFinite(pixel),
        ageAtRx: (Number.isFinite(pixel) ? pixel : 0) + (parseFloat(res.headers.get("X-Cache-Age-S")) || 0) + fetchS / 2,
        times: [...S.live.times, performance.now()].filter((x) => performance.now() - x < 10000) };
      if (!S.view) renderFrame();
      if (old) setTimeout(() => URL.revokeObjectURL(old), 2000);
    } catch (err) {
      S.live.err = err.message;
      renderFrame();
    }
    await sleep(Math.max(150, 1000 - (performance.now() - t0)));
  }
}

// One immediate look (workspace.look_now): stored as evidence, no task, no queue, no lease.
async function observeNow() {
  const d = dev();
  try {
    S.read = await post(`/api/devices/${encodeURIComponent(d.id)}/observe`, {});
  } catch (err) {
    if (err.status === 429) return toast(err.message || "It read this screen a moment ago. Try again in a few seconds.");
    throw err;
  }
  S.view = readView(S.read);
  if (S.mode !== "run") setMode("run");
  renderRead(); renderFrame(); renderStageFoot();
}

const readView = (r) => ({ url: r.frame_url, idx: -1, at: r.observed_at, label: `Screen read · ${hhmmss(r.observed_at)}`,
  caption: `Screen read at ${hhmmss(r.observed_at)}${r.pixel_age_ms != null ? ` · pixels ${(r.pixel_age_ms / 1000).toFixed(1)} s old when read` : ""}` });

function renderRead() {
  const r = S.read;
  readBox.hidden = !r || S.mode !== "run";
  if (!r) return mount(readBox);
  const lines = [...(r.items || [])].filter((i) => (i.text || "").trim()).sort((a, b) => (a.y - b.y) || (a.x - b.x));
  const show = S.readAll ? lines : lines.slice(0, 14);
  const mark = (i) => { S.view = { ...readView(r), pointPx: { x: i.x + (i.w || 0) / 2, y: i.y + (i.h || 0) / 2 } }; renderFrame(); renderStageFoot(); };
  mount(readBox,
    h("div", { class: "task-head" },
      h("div", { class: "grow" },
        h("h2", {}, "What's on the screen"),
        h("div", { class: "small muted meta" }, h("span", {}, hhmmss(r.observed_at)),
          r.foreground ? h("span", { title: r.foreground }, appName(r.foreground)) : null,
          r.screen ? h("span", {}, typeof r.screen === "string" ? r.screen : r.screen.title || r.screen.id || "") : null)),
      h("button", { class: "btn quiet sm", "aria-label": "Close the screen read", onclick: () => { S.read = null; S.view = null; renderRead(); renderFrame(); renderStageFoot(); } }, "Close")),
    h("div", { class: "read-body" },
      h("button", { class: "thumb", style: { width: "56px" }, "aria-label": "Show the frame it read",
        onclick: () => { S.view = readView(r); renderFrame(); renderStageFoot(); } },
        h("img", { src: r.frame_url, alt: "" })),
      lines.length ? h("ol", { class: "read-lines" }, show.map((i) => h("li", {}, h("button", { type: "button", onclick: () => mark(i), title: "Show where this is on the screen" }, i.text))))
        : h("p", { class: "muted small" }, "No text could be read on this screen.")),
    lines.length > show.length ? h("button", { class: "btn quiet sm", onclick: () => { S.readAll = true; renderRead(); } }, `Show all ${lines.length} lines`) : null,
    lines.length && navigator.clipboard ? h("button", { class: "btn quiet sm", onclick: () => navigator.clipboard.writeText(lines.map((i) => i.text).join("\n")).then(() => toast("Copied.")) }, "Copy text") : null);
}

// ---------- right panel: instruct, follow, stop ----------

const modebar = h("div", { class: "modebar", role: "tablist", "aria-label": "Panel" });
const composer = h("section", { class: "pane-sec composer" });
const readBox = h("section", { class: "pane-sec readbox", hidden: true });
const narrowTasks = h("section", { class: "pane-sec narrow-tasks" });
const taskBox = h("section", { class: "pane-sec taskbox", style: { borderBottom: 0, flex: 1 } });
const follow = h("form", { class: "follow", hidden: true });
const teachBox = h("section", { class: "pane-sec teach", hidden: true, style: { borderBottom: 0 } });
mount(panel, modebar, composer, readBox, taskBox, narrowTasks, follow, teachBox);

function renderMode() {
  const tab = (m, label) => h("button", { role: "tab", class: "mode", "aria-selected": String(S.mode === m), onclick: () => setMode(m) }, label);
  mount(modebar, tab("run", "Run"), tab("teach", "Teach"));
  for (const el of [composer, taskBox, narrowTasks]) el.hidden = S.mode !== "run";
  readBox.hidden = !S.read || S.mode !== "run";
  if (S.mode !== "run") follow.hidden = true;
  teachBox.hidden = S.mode !== "teach";
}

function setMode(m, opts) {
  S.mode = m;
  renderMode();
  if (m === "teach") openTeach(opts);
  else renderTask();
}

initTeach({
  box: teachBox,
  device: () => dev(),
  kinds: () => S.kinds,
  plain: (k) => plain(k),
  showFrame: (url, label, point) => { S.view = { url, idx: -1, at: null, label, point, caption: "Evidence frame from the recording" }; renderFrame(); renderStageFoot(); },
  runTask: async (kind, input) => {
    const t = await post("/api/tasks", { device_id: S.deviceId, kind, input });
    setMode("run");
    await refreshTasks();
    selectTask(t.id);
    toast(`${kindLabel(kind)} started.`);
  },
});

function plain(k) {
  const r = k.reason || "";
  if (/work session|lease|focus/i.test(r)) return "Needs a work session for this phone (lease and Focus) on this computer.";
  if (k.kind === "taught_run" && /routine/i.test(r)) return "Needs a taught routine you've confirmed.";
  if (/model|provider/i.test(r)) return "Needs a coach model connected on this computer.";
  if (/install|regist|isn't at|not found|no such file/i.test(r)) return "Not installed on this computer yet.";
  return r.replace(/\s*\([^)]*\)/g, "").trim() || "Not set up on this computer yet.";
}

function renderComposer() {
  const d = dev();
  const head = h("h2", {}, d ? `Ask ${d.name || d.id} to…` : "New task");
  if (S.problem) return mount(composer, head, h("p", { class: "callout" }, S.problem.message || S.problem.title));
  if (S.kindsProblem) return mount(composer, head, h("p", { class: "callout" }, S.kindsProblem));
  if (!S.kinds) return mount(composer, head, h("p", { class: "muted small" }, "Loading what this phone can do…"));
  // Viewing is never a task (root: a watch task would queue real work behind it); the screen is the viewport above.
  const kinds = S.kinds.filter((k) => k.kind !== "watch" && k.kind !== "observe");
  const ready = kinds.filter((k) => k.available);
  const blocked = kinds.filter((k) => !k.available);
  if (S.kind && (S.kind === "taught_run" || !ready.find((k) => k.kind === S.kind))) S.kind = null;

  const sel = ready.find((k) => k.kind === S.kind);
  const fields = Object.entries(sel?.input || {});
  const ex = (KIND[S.kind]?.examples || []).filter(([, v]) => Object.keys(v).every((f) => f in (sel?.input || {})));
  const offline = !d || d.connection?.state !== "connected";
  const form = h("form", { onsubmit: (e) => {
    e.preventDefault();
    busy(form.querySelector("button[type=submit]"), startTask);
  } },
  h("div", { class: "kinds", role: "radiogroup", "aria-label": "Task" }, kinds.map((k) => h("button", {
    type: "button", class: "kind", role: "radio", "aria-checked": String(k.kind === S.kind), disabled: !k.available,
    title: k.available ? KIND[k.kind]?.blurb || "" : plain(k),
    // a taught workflow is chosen from Saved workflows, whose card supplies its inputs: never a raw id or JSON here
    onclick: () => (k.kind === "taught_run" ? setMode("teach", { saved: true }) : (S.kind = k.kind, renderComposer())) },
    kindLabel(k.kind), k.kind === "taught_run" ? " →" : ""))),
  sel ? h("p", { class: "blurb" }, KIND[sel.kind]?.blurb || "") : h("p", { class: "blurb" }, ready.length ? "Choose what it should do." : "Nothing can run on this phone from here yet."),
  ex.length ? h("div", { class: "examples" }, ex.map(([label, vals]) => h("button", { type: "button", class: "example",
    onclick: () => { S.draft[S.kind] = { ...vals }; renderComposer(); } }, label))) : null,
  ...fieldViews(fields),
  sel ? h("button", { class: "btn primary", type: "submit", disabled: offline, title: offline ? d?.connection?.reason || "No phone connected." : "" },
    offline ? "Phone not connected" : "Start") : null,
  blocked.length ? h("details", { class: "setup" },
    h("summary", {}, `${blocked.length}${ready.length ? " more" : ""} need setup`),
    blocked.map((k) => h("p", {}, h("b", {}, kindLabel(k.kind)), " · ", plain(k))),
    h("details", { class: "raw" }, h("summary", {}, "Details"), blocked.map((k) => h("p", { class: "mono" }, `${k.kind}: ${k.reason || "—"}`)))) : null);
  mount(composer, head, form);
}

function fieldViews(fields) {
  const draft = (S.draft[S.kind] ||= {});
  const label = (f) => f.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
  const ticks = fields.filter(([, d]) => /^true:/i.test(d || ""));
  const choices = fields.filter(([, d]) => / \| /.test(d || "") && !/^optional:/i.test(d || ""));
  const optional = fields.filter(([, d]) => /^optional:/i.test(d || ""));
  const plainFields = fields.filter((x) => !ticks.includes(x) && !optional.includes(x) && !choices.includes(x));
  const choice = ([f, desc]) => {
    const opts = desc.split(" | ").map((o) => { const m = o.match(/^\s*([\w-]+)\s*(?:\((.*)\))?\s*$/); return m ? [m[1], (m[2] || "").replace(/^default:\s*/i, "")] : [o, ""]; });
    if (!(f in draft)) draft[f] = opts[0][0];
    return h("fieldset", { class: "choice-set" }, h("legend", {}, label(f)), opts.map(([v, why]) => h("label", { class: "check" },
      h("input", { type: "radio", name: `c-${S.kind}-${f}`, value: v, checked: draft[f] === v, onchange: () => (draft[f] = v) }),
      h("span", {}, h("b", {}, v.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase())), why ? h("span", { class: "muted" }, ` · ${why}`) : null))));
  };
  const text = ([f, desc]) => {
    const id = `f-${S.kind}-${f}`;
    const hint = (desc || "").replace(/^optional:\s*/i, "");
    const long = /request|note|text|instruction/i.test(f);
    const input = long ? h("textarea", { id, name: f, rows: 2, placeholder: hint }, draft[f] ?? "")
      : h("input", { id, name: f, type: /limit|count|max|minutes/i.test(f) ? "number" : "text", value: draft[f] ?? "", placeholder: hint });
    input.addEventListener("input", (e) => (draft[f] = e.target.value));
    return h("div", { class: "field" }, h("label", { for: id }, label(f)), input);
  };
  const tick = ([f, desc]) => {
    const box = h("input", { type: "checkbox", checked: draft[f] === true, onchange: (e) => (draft[f] = e.target.checked) });
    const words = (desc || "").replace(/^true:\s*/i, "");
    return h("label", { class: "check" }, box, h("span", {}, words.replace(/^./, (c) => c.toUpperCase())));
  };
  const notice = S.kind === "coach_observe" && ticks.length
    ? h("p", { class: "callout small" }, h("b", {}, "How it's processed: "), S.coachNote || "Loading the processing notice…") : null;
  if (S.kind === "coach_observe" && !S.coachNote) get("/api/coach/capability").then((c) => { S.coachNote = c.note; renderComposer(); }, () => {});
  return [...plainFields.map(text), ...choices.map(choice), notice, ...ticks.map(tick),
    optional.length ? h("details", { class: "more" }, h("summary", {}, `Add ${optional.map(([f]) => f.replace(/_/g, " ")).join(" or ")} (optional)`), optional.map(text)) : null];
}

async function startTask() {
  const k = S.kinds.find((x) => x.kind === S.kind);
  const input = {};
  for (const [f, desc] of Object.entries(k.input || {})) {
    if (/^true:/i.test(desc || "")) {
      if (S.draft[S.kind]?.[f] !== true) throw new Error(`Tick “${desc.replace(/^true:\s*/i, "").split(",")[0]}” first.`);
      input[f] = true;
      continue;
    }
    const v = (S.draft[S.kind]?.[f] ?? "").toString().trim();
    if (!v) continue;
    input[f] = /comma|list/i.test(desc || "") ? v.split(",").map((x) => x.trim().replace(/^@/, "")).filter(Boolean) : /limit|count|max|minutes/i.test(f) ? Number(v) : v;
  }
  const t = await post("/api/tasks", { device_id: S.deviceId, kind: S.kind, input });
  await refreshTasks();
  selectTask(t.id);
  toast(`${kindLabel(t.kind)} started.`);
}

function renderNarrowTasks() {
  mount(narrowTasks, h("h2", {}, "Tasks on this phone"), taskRows());
}

const isId = (k, v) => /(^|_)id$/.test(k) || (typeof v === "string" && /^[a-z]{3}_[0-9a-f]{12}$/.test(v));
const words = (k) => k.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
function shown(k, v) {
  if (v === true) return "Yes";
  if (v === false) return "No";
  if (Array.isArray(v)) return v.map((x) => (k === "handles" ? `@${String(x).replace(/^@/, "")}` : shown("", x))).join(", ");
  if (v && typeof v === "object") return Object.entries(v).map(([a, b]) => `${words(a)}: ${shown(a, b)}`).join(" · ");
  return typeof v === "string" && /^[a-z]+(_[a-z]+)+$/.test(v) ? words(v) : String(v);
}
// What the person asked for, in words: a workflow by its name, its params as their own fields; ids only under Details.
function inputSummary(t) {
  const out = [], ids = [];
  for (const [k, v] of Object.entries(t.input || {})) {
    if (v === "" || v == null || (Array.isArray(v) && !v.length)) continue;
    if (k === "routine_id") { out.push(["Workflow", S.names[v] || "a saved workflow"]); ids.push([k, v]); continue; }
    if ((k === "params" || k === "inputs") && v && typeof v === "object" && !Array.isArray(v)) {
      for (const [a, b] of Object.entries(v)) out.push([words(a), a === "handle" ? `@${String(b).replace(/^@/, "")}` : shown(a, b)]);
      continue;
    }
    if (isId(k, v)) { ids.push([k, v]); continue; }
    out.push([words(k), shown(k, v)]);
  }
  return { rows: out, ids };
}
// workflow names for routine ids seen in tasks (looked up once)
async function nameRoutine(id) {
  if (!id || id in S.names) return;
  S.names[id] = null;
  for (const url of [`/api/teach/routines/${encodeURIComponent(id)}`, `/api/routines/${encodeURIComponent(id)}`]) {
    try { S.names[id] = (await get(url)).name; break; } catch {}
  }
  if (S.names[id]) { renderSide(); renderNarrowTasks(); if (S.task?.input?.routine_id === id) renderTask(); }
}

function renderTask() {
  const t = S.task;
  if (!t) {
    mount(taskBox, h("p", { class: "muted" }, S.tasks.length
      ? "Pick a task to see what it found and each step it took on the phone."
      : "When a task runs, its answer appears here, with every step it took on the phone and the frame it saw."));
    follow.hidden = true;
    return;
  }
  const ex = t.execution || {};
  const open = OPEN.includes(t.state);
  const steps = t.steps || [];
  const { rows: asked, ids } = inputSummary(t);
  mount(taskBox,
    h("div", { class: "task-head" },
      h("div", { class: "grow" },
        h("h2", {}, kindLabel(t.kind)),
        h("div", { class: "small muted meta" },
          h("span", { class: `state ${t.state}` }, STATE[t.state] || t.state),
          h("span", {}, `${hhmmss(t.started || t.created)}${t.finished ? `–${hhmmss(t.finished)}` : ""}`),
          t.origin && t.origin !== "operator" ? h("span", { class: "tag" }, t.origin === "agent" ? "Started by an agent" : "Test") : null,
          ex.simulated ? h("span", { class: "tag sim" }, "Simulated, not the phone") : null)),
      open ? stopControls(t, { takeover: false })[0] : null),
    asked.length ? h("dl", { class: "kv asked" }, asked.map(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])) : null,
    ids.length ? h("details", { class: "raw-step" }, h("summary", {}, "Details"), h("pre", {}, ids.map(([k, v]) => `${k}: ${v}`).join("\n"))) : null,
    t.state === "needs" && t.needs ? h("p", { class: `callout ${t.kind === "coach_observe" ? "info" : "warn"}` },
      t.kind === "coach_observe" ? null : h("b", {}, "It needs you: "), t.needs) : null,
    t.error ? h("p", { class: "callout bad" }, t.error) : null,
    t.result ? h("div", { class: "answer" }, h("div", { class: "label" }, "What it found"),
      t.result.reads && typeof t.result.reads === "object" ? profileCard(t) : resultView(t.result))
      : open && t.kind !== "coach_observe" ? h("p", { class: "muted small" }, steps.length ? `Working… ${steps.length} ${steps.length === 1 ? "step" : "steps"} so far.` : "Starting…") : null,
    open && dev()?.task_id !== t.id && !releasing(t) ? h("p", { class: "small" },
      h("button", { class: "btn", onclick: (e) => busy(e.currentTarget, () => act(t.id, "handoff", { note: "" })) }, "Take over the phone")) : null,
    releasing(t) ? h("p", { class: "callout info" }, releasing(t) === "handoff"
      ? "Handing the phone back. The step already under way finishes first; then the task lets go of the phone."
      : "Stopping. The step already under way finishes first; nothing new starts.") : null,
    t.kind === "coach_observe" ? chatView(t) : null,
    t.kind === "coach_observe" ? coachControls(t) : null,
    // do it once here, then teach it: a finished task that kept its frames is a demonstration
    // (a run on a simulated phone is never a demonstration: teaching won't list it, so the button isn't offered)
    !open && t.kind !== "coach_observe" && !ex.simulated && steps.some((x) => x.frame_id) ? h("div", { class: "row-actions" },
      h("button", { class: "btn sm", onclick: () => setMode("teach", { sessionId: t.id }) }, "Teach a workflow from this task")) : null,
    steps.length && t.kind !== "coach_observe" ? h("details", { class: "steps-box", open: !!S.stepsOpen[t.id],
      ontoggle: (e) => (S.stepsOpen[t.id] = e.currentTarget.open) },
      h("summary", {}, `What it did · ${steps.length} ${steps.length === 1 ? "step" : "steps"}`),
      h("ol", { class: "steps" }, steps.map(stepRow))) : null);
  renderFollow(t);
}

// The coach as a conversation (phone_coach.py): its reads of the phone, your words, its answers (result steps with
// role answer|review|sent, each with the screen it rests on and the messages it quotes), drafts it put on the phone,
// and the exact-text approval for sending. Older coach builds logged answers as message steps with evidence.
const isAnswer = (x) => (x.kind === "result" && x.data?.role) || (x.kind === "message" && (x.data?.evidence || x.data?.review_id || x.data?.thread_id));
const threadOf = (t) => (t.steps || []).map((x) => x.data?.thread_id).find(Boolean) || null;
const say = (t, text) => post(`/api/tasks/${t.id}/message`, { text }).then((nt) => { S.task = { ...nt, steps: S.task?.steps || [] }; renderTask(); });

function chatView(t) {
  const steps = t.steps || [];
  const mine = steps.filter((x) => x.kind === "message" && !isAnswer(x));
  const items = [
    ...(mine.length ? [] : (t.messages || []).map((m) => ({ at: m.at, who: "you", text: m.text }))),
    ...steps.map((x) => ({ at: x.at, idx: x.idx, who: x.kind === "message" && !isAnswer(x) ? "you" : isAnswer(x) ? "coach" : "phone", text: x.text, step: x })),
  ].sort((a, b) => String(a.at).localeCompare(String(b.at)) || (a.idx ?? 0) - (b.idx ?? 0));
  if (!items.length) return null;
  const open = OPEN.includes(t.state);
  const mode = t.input?.mode || "review_only";
  const thumb = (x, label) => (x.frame_url ? h("button", { class: "thumb", "aria-label": label,
    onclick: () => { S.view = { url: x.frame_url, idx: x.idx, at: x.at }; renderFrame(); renderStageFoot(); } }, h("img", { src: x.frame_url, alt: "", loading: "lazy" })) : null);
  return h("div", { class: "chat" }, items.map((it) => {
    if (it.who === "you") return h("div", { class: "bubble you" }, it.text);
    const x = it.step;
    const d = x.data || {};
    if (it.who === "phone") {
      const code = d.draft_code;
      const { line, raw } = stepLine(x.text);
      return h("div", { class: "chat-note" }, svg(x.kind), h("span", { class: "grow" }, line,
        raw ? h("details", { class: "raw-step" }, h("summary", {}, "Details"), h("pre", {}, raw)) : null,
        code && open ? h("div", { class: "approve" },
          h("span", { class: "small" }, "Send exactly what's in the message box on the phone?"),
          h("button", { class: "btn sm primary", onclick: (e) => busy(e.currentTarget, () => say(t, `send ${code}`)) }, "Approve and send")) : null),
        thumb(x, "Show this screen"));
    }
    // evidence: the observed messages an answer rests on ({message, who, text, frames}), each linked to its frame
    const ev = [].concat(d.evidence || []).map((e) => (typeof e === "string" ? { text: e } : e)).filter((e) => e.text || e.quote);
    const quote = (e) => h("q", {}, e.who ? h("b", {}, e.who === "me" ? "You: " : "Them: ") : null, e.text || e.quote,
      e.frames?.[0] ? h("button", { class: "linkish", type: "button", onclick: () => { S.view = { url: `/api/frames/${e.frames[0]}`, idx: -1, at: e.first_seen, label: "Where this message was read", caption: "The screen this message was read from" }; renderFrame(); renderStageFoot(); } }, " see it") : null);
    return h("div", { class: `bubble coach ${d.role === "sent" ? "sent" : ""}` },
      d.role === "review" ? h("div", { class: "label" }, "Review") : d.role === "sent" ? h("div", { class: "label" }, d.verified ? "Sent · seen on the phone" : "Sent") : null,
      x.text,
      ev.length ? h("div", { class: "quotes" }, ev.slice(0, 3).map(quote)) : null,
      d.role === "review" ? reviewDetail(d) : null,
      h("div", { class: "bubble-foot" },
        thumb(x, "Show the screen this rests on"),
        d.review_id ? h("a", { href: `/review/#/r/${encodeURIComponent(d.review_id)}`, target: "_blank", rel: "noopener" }, "Open the message-by-message review") : null),
      d.draft ? h("div", { class: "draft" }, h("div", { class: "label" }, "Suggested reply · not sent"), h("p", {}, d.draft),
        d.draft_why ? h("p", { class: "small muted" }, d.draft_why) : null,
        h("div", { class: "row-actions", style: { marginTop: "4px" } },
          open && mode !== "review_only" ? h("button", { class: "btn sm", onclick: (e) => busy(e.currentTarget, () => say(t, "type it")) }, "Type it on the phone") : null,
          navigator.clipboard ? h("button", { class: "btn sm quiet", type: "button", onclick: () => navigator.clipboard.writeText(d.draft).then(() => toast("Copied.")) }, "Copy") : null)) : null);
  }));
}

// A review the person asked for: each of their messages with its rating and the reason, then the lessons.
function reviewDetail(d) {
  const items = d.items || [];
  if (!items.length && !(d.lessons || []).length) return null;
  const RATE = { strong: "Strong", good: "Good", mixed: "Mixed", weak: "Weak", rethink: "Rethink" };
  return h("div", { class: "review" },
    items.map((i) => h("div", { class: "review-item" },
      h("div", { class: "row-actions" }, i.rating ? h("span", { class: `rate ${i.rating}` }, RATE[i.rating] || i.rating) : null, h("span", { class: "small" }, i.text)),
      i.reason ? h("p", { class: "small muted" }, i.reason) : null,
      i.alternative_kind === "rewrite" || i.alternative_kind === "optional" ? (i.alternative ? h("p", { class: "small" }, h("b", {}, i.alternative_kind === "optional" ? "Optional: " : "Try: "), i.alternative) : null)
        : i.alternative_kind === "send_nothing" ? h("p", { class: "small" }, h("b", {}, "Better: send nothing.")) : null)),
    (d.lessons || []).length ? h("div", { class: "lessons" }, h("div", { class: "label" }, "Lessons"),
      d.lessons.map((l) => h("p", { class: "small" }, h("b", {}, l.title), l.why ? ` · ${l.why}` : "", l.practice ? h("span", { class: "muted" }, ` Practice: ${l.practice}`) : null))) : null);
}

function coachControls(t) {
  const open = OPEN.includes(t.state);
  const tid = threadOf(t);
  if (open) return h("div", { class: "row-actions" },
    h("button", { class: "btn sm", onclick: (e) => busy(e.currentTarget, () => say(t, "review")) }, "Review my messages"),
    h("button", { class: "btn sm", onclick: (e) => busy(e.currentTarget, () => say(t, "read again")) }, "Read the screen again"));
  if (!tid) return null;
  const wrap = h("div", { class: "row-actions" });
  const ask = () => mount(wrap, h("button", { class: "btn sm quiet danger-text", onclick: () => mount(wrap,
    h("span", { class: "small" }, "Delete the conversation it read, its answers, reviews and the frames it stored?"),
    h("button", { class: "btn sm stop", onclick: (e) => busy(e.currentTarget, async () => {
      const r = await api_delete(`/api/coach/phone/threads/${encodeURIComponent(tid)}`);
      mount(wrap, h("p", { class: `callout ${r.complete ? "info" : "warn"}` }, r.complete ? "Deleted, including the frames this task stored."
        : ["Deleted the conversation. Still stored: ", (r.remaining || []).join("; ")]));
      refreshTasks();
    }) }, "Delete"),
    h("button", { class: "btn sm quiet", onclick: ask }, "Cancel")) }, "Delete this conversation's data"));
  ask();
  return wrap;
}

// A step's line for a person; driver timings, exceptions and raw values go under Details, never in the line.
const GUARDS = [
  [/stale ?input|frame (is )?too old|older than the input|frame_age/i, "The phone refused an input based on an old screen; reading it again."],
  [/not ?frontmost|wasn't in front|not in front/i, "The app wasn't in front, so nothing was touched."],
  [/unfamiliar ?screen|didn't recognise|unknown screen/i, "It didn't recognise this screen, so it didn't touch it."],
  [/\bparked\b/i, "Paused: the phone isn't ready for input yet."],
  [/lock ?screen|is locked/i, "The phone is locked, so nothing was touched."],
];
function stepLine(text) {
  const t = String(text || "");
  const guard = GUARDS.find(([re]) => re.test(t));
  if (guard) return { line: guard[1], raw: t };
  // a taught run's step: "step 4: On <screen as read>, <what it does> (taught pause 2.357 s, drawn 3.1 s)".
  // The line keeps the step and what it does; the screen as read and the timings stay whole under Details.
  const taught = t.match(/^step\s*(\d+)\s*:\s*([\s\S]*)$/i);
  if (taught || /\((?:[^()]*\b(?:taught pause|drawn|pace|ms)\b[^()]*)\)\s*$/i.test(t)) {
    const what = (taught ? taught[2] : t)
      .replace(/\s*\((?:[^()]*\b(?:taught pause|drawn|pace|ms)\b[^()]*)\)\s*$/i, "")
      .replace(/^on\s[^,]{1,80},\s*/i, "").trim();
    if (what) return { line: `${taught ? `Step ${taught[1]} · ` : ""}${what[0].toUpperCase()}${what.slice(1)}`, raw: what === t ? null : t };
  }
  const technical = t.length > 160 || /Traceback|Exception|Error\(|\bms=|\{["']|\w+=\d/.test(t);
  if (!technical) return { line: t, raw: null };
  const head = t.split(/(?<=[.!?])\s|:\s|\s\(|\n/)[0].replace(/\b\w+(Error|Exception)\b:?/g, "").trim();
  return { line: head && head.length <= 160 ? head : "Something on the phone didn't go as expected.", raw: t };
}

function stepRow(s) {
  const point = s.data && s.data.x != null && s.data.y != null && s.data.width && s.data.height
    ? { x: s.data.x / s.data.width, y: s.data.y / s.data.height } : null;
  return h("li", { class: `step ${s.kind}` },
    h("span", { class: "at" }, hhmmss(s.at)),
    svg(s.kind),
    (() => {
      const { line, raw } = stepLine(s.text || s.kind);
      return h("span", { class: "txt" }, line, s.pace_ms != null ? h("span", { class: "pace" }, ` · after ${(s.pace_ms / 1000).toFixed(1)} s`) : null,
        raw ? h("details", { class: "raw-step" }, h("summary", {}, "Details"), h("pre", {}, raw)) : null);
    })(),
    s.frame_url ? h("button", { class: "thumb", "aria-pressed": String(S.view?.idx === s.idx), "aria-label": `Show the frame from step ${s.idx + 1}`,
      onclick: () => { S.view = S.view?.idx === s.idx ? null : { url: s.frame_url, idx: s.idx, at: s.at, point }; renderFrame(); renderStageFoot(); renderTask(); } },
      h("img", { src: s.frame_url, alt: "", loading: "lazy" })) : h("span", {}));
}

// A profile read as a person reads it: the handle, the counts exactly as the screen displayed them, when, and the
// one frame that shows them. Used for taught runs (and any result with the same reads shape).
const METRIC = { posts: "Posts", followers: "Followers", following: "Following" };
function profileCard(t) {
  const r = t.result || {};
  const reads = r.reads || {};
  const handle = String(r.params?.handle || t.input?.params?.handle || t.input?.inputs?.handle || "").replace(/^@/, "");
  const frame = r.result_frame || Object.values(reads).find((x) => x?.frame_id) || {};
  const phases = (t.steps || []).filter((x) => x.kind === "tap" || x.kind === "type").map((x) => x.data?.phase);
  const taught = phases.filter((x) => x === "taught").length || r.learned_steps_used || r.steps_by_tier?.taught || 0;
  const start = phases.filter((x) => x === "start").length;
  const name = r.routine?.id ? S.names[r.routine.id] : null;
  if (r.routine?.id) nameRoutine(r.routine.id);
  const details = {
    workflow: r.routine ? `${name || "a saved workflow"} · version ${r.routine.version} · ${r.routine.provenance || ""} · ${r.routine.id}` : null,
    limits: r.limits || [],
    pace: (r.pace || []).map((x) => `step ${x.step}: taught ${x.taught_ms} ms, drawn ${x.drawn_ms} ms`),
  };
  return h("div", { class: "card profile" },
    h("div", { class: "profile-head" },
      handle ? h("b", {}, `@${handle}`) : null,
      r.handle_shown ? h("span", { class: "small muted", title: "The handle on screen matched the one asked for" }, "handle seen on the profile") : null),
    Object.keys(reads).length ? h("div", { class: "metrics" }, Object.entries(reads).map(([k, v]) => h("div", { class: "metric" },
      h("span", { class: "metric-value num" }, v?.value ?? "—"), h("span", { class: "metric-label" }, METRIC[k] || words(k))))) : null,
    h("div", { class: "row-actions" },
      frame.frame_id ? h("button", { class: "btn sm", onclick: () => { S.view = { url: `/api/frames/${frame.frame_id}`, idx: -1, at: frame.observed_at,
        label: `Evidence · ${hhmmss(frame.observed_at)}`, caption: `The profile as read at ${hhmmss(frame.observed_at)}` }; renderFrame(); renderStageFoot(); } }, svg("observe"), "View evidence") : null,
      frame.observed_at ? h("span", { class: "small muted" }, `Seen ${hhmmss(frame.observed_at)}`) : null),
    taught || start ? h("p", { class: "small muted reuse" }, `Reused ${name ? `“${name}”` : "your taught workflow"}: ${taught} taught ${taught === 1 ? "input" : "inputs"}`,
      start ? `, ${start} to reach its start screen` : "", ".") : null,
    h("details", { class: "raw-step" }, h("summary", {}, "Details"),
      h("pre", {}, [details.workflow, ...details.limits.map((x) => `limit: ${x}`), ...details.pace].filter(Boolean).join("\n")),
      h("pre", {}, JSON.stringify(r, null, 1))));
}

// Results are whatever the executor proved; every claim resting on the screen carries its frame and time.
function resultView(v, depth = 0) {
  if (v == null || v === "") return null;
  if (Array.isArray(v)) {
    if (!v.length) return h("span", { class: "muted" }, "none");
    if (v.every((x) => typeof x !== "object")) return h("span", {}, v.join(", "));
    return h("div", { class: "cards" }, v.map((x) => resultView(x, depth + 1)));
  }
  if (typeof v === "object" && depth >= 1 && Object.values(v).every((x) => x == null || typeof x !== "object")) {
    return h("span", {}, Object.entries(v).filter(([k, x]) => x != null && x !== "" && !/(^|_)id$|_at$/.test(k)).map(([k, x]) => `${words(k)} ${shown(k, x)}`).join(" · "));
  }
  if (typeof v === "object") {
    const ev = v.frame_id || v.frame_url ? h("button", { class: "evidence btn quiet", style: { height: "auto", padding: 0 },
      onclick: () => { S.view = { url: v.frame_url || `/api/frames/${v.frame_id}`, idx: -1, at: v.observed_at }; renderFrame(); renderStageFoot(); } },
      h("span", { class: "thumb", style: { width: "22px" } }, h("img", { src: v.frame_url || `/api/frames/${v.frame_id}`, alt: "" })),
      `Seen ${hhmmss(v.observed_at)}`) : null;
    const rows = Object.entries(v).filter(([k]) => !["frame_id", "frame_url", "observed_at"].includes(k));
    const body = h("dl", { class: "kv" }, rows.map(([k, x]) => [h("dt", {}, k.replace(/_/g, " ")), h("dd", {}, resultView(x, depth + 1))]));
    return depth ? h("div", { class: "card" }, body, ev) : h("div", { class: "card" }, body, ev);
  }
  const s = String(v);
  return /^https?:\/\//.test(s) ? h("a", { href: s, target: "_blank", rel: "noopener" }, s) : h("span", {}, s);
}

function renderFollow(t) {
  const open = OPEN.includes(t.state);
  follow.hidden = !open || S.mode !== "run";
  if (!open) return;
  const input = h("input", { type: "text", "aria-label": "Message the task",
    placeholder: t.kind === "coach_observe" ? "Ask about this conversation…" : t.state === "needs" ? "Answer it…" : "Refine or ask a follow-up…" });
  follow.onsubmit = (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    busy(follow.querySelector("button"), async () => {
      S.task = await post(`/api/tasks/${t.id}/message`, { text });
      input.value = "";
      renderTask();
    });
  };
  mount(follow, input, h("button", { class: "btn primary", type: "submit" }, "Send"));
}

async function act(id, what, body = {}) {
  const t = await post(`/api/tasks/${id}/${what}`, body);
  agree(t);
  if (OPEN.includes(t.state)) {
    S.releasing[id] = what;
    toast(what === "stop" ? "Stopping: the step already under way finishes first." : "Handing the phone back: the step already under way finishes, then it lets go.");
  } else {
    toast(ended(t));
  }
  if (S.taskId !== id) selectTask(id); // watch it until it really ends
  else S.task = { ...t, steps: S.task?.steps || t.steps || [] };
  await refreshTasks();
  renderTask();
  renderStageFoot();
}

// One task's newest state, applied everywhere it shows: its sidebar row, and the phone's "working on" under the stage.
function agree(t) {
  const row = S.tasks.find((x) => x.id === t.id);
  if (row) Object.assign(row, { state: t.state, stop_requested: t.stop_requested, finished: t.finished, needs: t.needs });
  const d = S.devices?.find((x) => x.id === t.device_id);
  if (d && !OPEN.includes(t.state) && d.task_id === t.id) d.task_id = null;
  if (d && OPEN.includes(t.state) && !d.task_id) d.task_id = t.id;
  renderSide(); renderNarrowTasks(); renderStageFoot();
}

const ended = (t) => (t.state === "handed_off" ? "The phone is yours now: the task has let go of it."
  : t.state === "stopped" ? "Stopped. Nothing more will happen on the phone." : `The task ended: ${(STATE[t.state] || t.state).toLowerCase()}.`);
const releasing = (t) => (t && OPEN.includes(t.state) && (S.releasing[t.id] || (t.stop_requested ? "stop" : null))) || null;

// Stop and Take over, or what they're waiting for: never offered twice while a release is under way.
function stopControls(t, { takeover = true } = {}) {
  const r = releasing(t);
  if (r) return [h("button", { class: "btn stop", disabled: true }, r === "handoff" ? "Handing back…" : "Stopping…")];
  return [
    h("button", { class: "btn stop", onclick: (e) => busy(e.currentTarget, () => act(t.id, "stop")) }, "Stop"),
    takeover ? h("button", { class: "btn", onclick: (e) => busy(e.currentTarget, () => act(t.id, "handoff", { note: "" })) }, "Take over the phone") : null,
  ];
}

// ---------- loading and polling ----------

function problemFrom(err) {
  if (err instanceof ApiError && (err.status === 401 || err.status === 403)) return { kind: "auth" };
  if (err instanceof ApiError && err.status === 404)
    return { kind: "missing", short: "Workspace service not running", title: "The workspace service isn't running here",
      message: "This server doesn't answer /api/devices yet, so nothing here is live." };
  return { kind: "error", short: "Can't reach the server", title: "Can't reach the server", message: err.message };
}

function renderSignIn() {
  document.body.replaceChildren(h("div", { class: "full" }, h("div", { class: "box" },
    h("div", { class: "brand", style: { padding: 0 } }, h("span", { class: "brand-mark", "aria-hidden": "true" }), "Phonebridge"),
    h("h1", {}, "Operator sign-in"),
    h("p", { class: "muted" }, "This workspace shows the operator's own phones. Open your operator sign-in link on this computer; it keeps you signed in here."),
    S.hasReview ? h("p", { class: "small muted" }, "Reviewing a dating conversation instead? ", h("a", { href: "/review/" }, "Game Review")) : null)));
}

// One poll of each kind in flight at a time: a slow answer must not stack up requests behind it.
const once = (fn) => { let busy = null; return () => (busy ||= fn().finally(() => (busy = null))); };

const refreshDevices = once(async () => {
  try {
    S.devices = await get("/api/devices");
    S.problem = null;
  } catch (err) {
    S.problem = problemFrom(err);
    if (S.problem.kind === "auth") return renderSignIn(), false;
  }
  if (S.devices?.length && !dev()) S.deviceId = S.devices[0].id;
  renderSide();
  renderStageHead();
  renderStageFoot();
  renderFrame();
  teachStateChanged();
  return true;
});

async function refreshKinds() {
  try {
    const k = await get("/api/tasks/kinds");
    S.kindsProblem = null;
    const key = JSON.stringify(k);
    if (key !== S.kindsKey) { S.kinds = k; S.kindsKey = key; renderComposer(); teachStateChanged(); }
  } catch (err) {
    S.kindsProblem = err.status === 404 ? "This server can't run phone tasks yet (no /api/tasks/kinds)." : err.message;
    renderComposer();
  }
}

const refreshTasks = once(async () => {
  if (!S.deviceId || S.problem) return;
  try {
    S.tasks = await get(`/api/tasks?device_id=${encodeURIComponent(S.deviceId)}`);
  } catch {
    S.tasks = [];
  }
  if (!S.taskId && S.tasks.length) S.taskId = S.tasks[0].id, taskLoop(S.taskId);
  const row = S.tasks.find((x) => x.id === S.taskId);
  if (row && S.task?.id === row.id && OPEN.includes(S.task.state) && !OPEN.includes(row.state)) taskLoop(row.id);
  renderSide();
  renderNarrowTasks();
  renderStageFoot();
});

function selectDevice(id) {
  if (id === S.deviceId) return;
  S.deviceId = id;
  S.taskId = null;
  S.task = null;
  S.view = null;
  S.tasks = [];
  S.live = { url: null, ageAtRx: null, rxAt: 0, capturedAt: null, source: null, err: null, times: [] };
  renderSide(); renderStageHead(); renderStageFoot(); renderFrame(); renderComposer(); renderTask();
  refreshTasks();
}

function selectTask(id) {
  S.taskId = id;
  S.view = null;
  S.read = null;
  renderRead();
  renderSide();
  renderNarrowTasks();
  taskLoop(id);
}

// Long-poll one task: new steps or a state change come back as soon as they happen.
let loopToken = 0;
async function taskLoop(id) {
  const mine = ++loopToken;
  let after = -1;
  let settling = false; // the final "Done" step is in, the state isn't yet: a long wait could miss it
  if (S.task?.id !== id) S.task = null;
  for (;;) {
    if (mine !== loopToken) return;
    try {
      // workspace.wait_task wakes on a change from the state it began with; a task that ended between two polls
      // looks unchanged to it, so never long-poll once the run has written its final step
      const t = await get(`/api/tasks/${encodeURIComponent(id)}?after=${after}&wait=${visible() && !settling ? 20 : 0}`);
      if (mine !== loopToken) return;
      const byIdx = new Map((after < 0 ? [] : S.task?.steps || []).map((x) => [x.idx, x]));
      for (const x of t.steps || []) byIdx.set(x.idx, x); // a repeated step replaces itself, never duplicates
      const steps = [...byIdx.values()].sort((a, b) => a.idx - b.idx);
      const prev = S.task;
      S.task = { ...t, steps };
      after = steps.length ? steps.at(-1).idx : after;
      // redraw only on change, so an open inline form or a hovered control survives a quiet poll
      const changed = !prev || prev.state !== t.state || prev.steps.length !== steps.length || prev.needs !== t.needs ||
        JSON.stringify(prev.result) !== JSON.stringify(t.result) || (prev.messages || []).length !== (t.messages || []).length;
      if (changed) agree(t);
      if (changed && S.releasing[t.id] && !OPEN.includes(t.state)) {
        toast(ended(t));
        delete S.releasing[t.id];
        refreshTasks();
      }
      if (changed) { renderTask(); renderStageFoot(); }
      if (!OPEN.includes(t.state)) return;
      settling = steps.at(-1)?.kind === "result";
      if (settling) await sleep(300);
      else if (!visible()) await sleep(3000);
      else if (!changed) await sleep(1000); // a server that answers at once without news is not polled in a loop
    } catch (err) {
      if (mine !== loopToken) return;
      mount(taskBox, h("p", { class: "callout bad" }, err.message));
      await sleep(4000);
    }
  }
}

// Pages beyond the workspace exist only in some deployments (the standalone release ships the workspace alone):
// link to them only when this server serves them.
const served = (path) => fetch(path, { method: "HEAD", credentials: "same-origin" }).then((r) => r.ok, () => false);
async function findExtras() {
  const pages = [["/jobs/", "Jobs and approvals"], ["/ops/", "Operations"]];
  const ok = await Promise.all(pages.map(([p]) => served(p)));
  S.extras = pages.filter((_, i) => ok[i]);
  renderSide();
}

async function boot() {
  renderMode(); renderSide(); renderStageHead(); renderComposer(); renderTask(); renderFrame();
  S.hasReview = await served("/review/");
  if (!(await refreshDevices())) return;
  findExtras();
  await Promise.all([refreshKinds(), refreshTasks()]);
  renderComposer();
  frameLoop();
  setInterval(() => visible() && refreshDevices(), 4000);
  setInterval(() => visible() && refreshTasks(), 4000);
  setInterval(() => visible() && refreshKinds(), 30000);
  setInterval(() => !S.view && renderFrame(), 500);
  document.addEventListener("visibilitychange", () => { renderFrame(); if (visible()) refreshDevices(); });
}
boot();
