// Teach a workflow from the workspace: a real recording (uploaded, or a captured phone session) and the person's own
// description → the steps, inputs and rules teaching.py derived, each with its evidence → confirm → run on fresh inputs.
// Routes: teaching.py (/api/teach/*), agency routines (/api/routines/*), the workspace's POST /api/tasks (taught_run).
import { get, post, upload, ApiError } from "./api.js";
import { h, mount, toast, busy, when, plural } from "./ui.js";

const T = { routines: null, problem: null, routine: null, sessions: null, sessionsProblem: null, source: "upload", file: null,
  name: "", narration: "", externalOk: false, inputs: {}, learning: false };

let ctx; // {box, device(), kinds(), plain(kind), showFrame(url, label), runTask(kind, input)}

export function initTeach(c) {
  ctx = c;
}

// opts.sessionId: open Teach on that captured session (a finished workspace task's id from the Run tab)
export async function openTeach(opts = {}) {
  T.saved = !!opts.saved; // came to run a saved workflow: those first, teaching a new one second
  if (opts.sessionId) {
    T.routine = null;
    T.source = "session";
    T.session = opts.sessionId;
    T.first = T.last = null;
  }
  render();
  try {
    T.routines = await get("/api/routines");
    T.problem = null;
  } catch (err) {
    T.problem = err instanceof ApiError && (err.status === 401 || err.status === 403)
      ? "Teaching saves recordings to your workspace, so it needs your founder sign-in link (the operator token can't own recordings)."
      : err.message;
  }
  try {
    T.sessions = await get("/api/teach/sessions");
    T.sessionsProblem = null;
  } catch (err) {
    T.sessions = null;
    T.sessionsProblem = err.status === 404 ? "Teaching from a captured phone session isn't installed on this computer yet." : err.message;
  }
  if (T.session && T.source === "session") loadSteps(T.session);
  render();
}

// A workspace task as the demonstration: its steps, so the person can keep only the part to teach.
async function loadSteps(id) {
  if (!/^tsk_/.test(id || "") || T.steps?.id === id) return;
  try {
    const t = await get(`/api/tasks/${encodeURIComponent(id)}`);
    T.steps = { id, list: t.steps || [] };
  } catch {
    T.steps = { id, list: [] };
  }
  render();
}

const stepText = (s) => {
  if (typeof s === "string") return s;
  return s.text || [s.op, s.label && `“${s.label}”`].filter(Boolean).join(" ") || JSON.stringify(s);
};
// The screen the step started from (teaching.py: step.screen.frame, else a proof frame, else the screen after it),
// and where the touch landed when the recording shows it (proof kind "touch", at = [fx, fy]).
const evidence = (s) => {
  const proof = s.proof || [];
  const frame = s.screen?.frame || proof.find((p) => p.frame)?.frame || s.expect?.frame;
  const touch = proof.find((p) => p.kind === "touch" && Array.isArray(p.at));
  return frame ? { frame, point: touch ? { x: touch.at[0], y: touch.at[1] } : null } : null;
};

function render() {
  const box = ctx.box;
  if (T.problem) return mount(box, h("p", { class: "callout" }, T.problem));
  if (T.routine) return mount(box, review());
  mount(box, ...(T.saved ? [savedList(true), newForm(true)] : [newForm(), savedList()]));
}

// Saved workflows (what can run) are kept apart from the recordings and runs they were taught from (the source).
function savedList(first = false) {
  const list = T.routines || [];
  return h("div", { class: `teach-saved ${first ? "first" : ""}` },
    first ? h("h2", {}, "Run a saved workflow") : h("div", { class: "label" }, "Saved workflows"),
    first ? h("p", { class: "blurb" }, "Pick one; its page asks for the inputs that change each time, then runs it on the phone.") : null,
    !T.routines ? h("p", { class: "muted small" }, "Loading…")
      : !list.length ? h("p", { class: "muted small" }, "None yet. The first one comes from a recording of you doing the job once.")
      : list.map((r) => h("button", { class: "row-btn", onclick: (e) => busy(e.currentTarget, () => openRoutine(r.id)) },
          h("span", { class: "grow" },
            h("div", { class: "t" }, r.name),
            h("div", { class: "s" }, [r.confirmed_at ? "Confirmed" : "Not confirmed yet", r.version ? `version ${r.version}` : null, when(r.created)].filter(Boolean).join(" · "))))));
}

function newForm(second = false) {
  const sessionsOk = !T.sessionsProblem && T.sessions;
  const fileIn = h("input", { type: "file", accept: "video/*", id: "teach-file", onchange: (e) => { T.file = e.target.files[0] || null; render(); } });
  const name = h("input", { type: "text", id: "teach-name", value: T.name, placeholder: "e.g. Check a creator's profile", oninput: (e) => (T.name = e.target.value) });
  const narr = h("textarea", { id: "teach-narr", rows: 3, placeholder: "What the job is, what it should always do, and what changes each time (e.g. the handle).",
    oninput: (e) => (T.narration = e.target.value) }, T.narration);
  const ext = h("input", { type: "checkbox", checked: T.externalOk, onchange: (e) => (T.externalOk = e.target.checked) });
  const sessionSel = sessionsOk && T.sessions.length ? h("select", { id: "teach-session", onchange: (e) => { T.session = e.target.value; T.first = T.last = null; loadSteps(T.session); render(); } },
    T.sessions.map((s) => h("option", { value: s.id, selected: s.id === T.session },
      `${s.label || s.id} · ${when(s.captured_at)}${s.duration_s ? ` · ${Math.round(s.duration_s)} s` : ""}${s.kind === "workspace task" ? " · from this workspace" : ""}`))) : null;
  if (sessionSel && !T.session) { T.session = T.sessions[0].id; loadSteps(T.session); }
  const steps = T.session && T.steps?.id === T.session ? T.steps.list : [];
  const stepPick = (label, key, dflt) => h("label", { class: "range" }, label, h("select", { onchange: (e) => (T[key] = e.target.value === "" ? null : Number(e.target.value)) },
    h("option", { value: "" }, dflt), steps.map((x) => h("option", { value: x.idx, selected: T[key] === x.idx }, `${x.idx + 1}. ${(x.text || x.kind).slice(0, 60)}`))));
  const missing = T.source === "session" && T.session && sessionsOk && !T.sessions.some((x) => x.id === T.session)
    ? h("p", { class: "callout" }, /^tsk_/.test(T.session)
      ? "This run isn't one you can teach from on this computer (teaching from workspace runs may not be installed yet)."
      : "That captured session isn't available here.") : null;
  const range = T.source === "session" && steps.length > 1 ? h("div", { class: "field" }, h("span", { class: "hint" }, "Teach only part of it (optional)"),
    h("div", { class: "row-actions" }, stepPick("From ", "first", "the first step"), stepPick("To ", "last", "the last step"))) : null;
  const ready = T.source === "upload" ? !!T.file : !!(sessionsOk && T.session && T.sessions.some((x) => x.id === T.session));
  const form = h("form", { class: "teach-new", onsubmit: (e) => { e.preventDefault(); busy(form.querySelector("button[type=submit]"), learn); } },
    h("h2", { style: second ? { marginTop: "26px" } : null }, second ? "Or teach a new one" : "Teach a workflow"),
    h("p", { class: "blurb" }, "Do the job once on the phone while it records, then describe it. It works out the steps and what changes each time; you check them before it ever runs."),
    h("div", { class: "kinds", role: "radiogroup", "aria-label": "Recording" },
      h("button", { type: "button", class: "kind", role: "radio", "aria-checked": String(T.source === "upload"), onclick: () => { T.source = "upload"; render(); } }, "Upload a screen recording"),
      h("button", { type: "button", class: "kind", role: "radio", "aria-checked": String(T.source === "session"), disabled: !sessionsOk || !T.sessions.length,
        title: T.sessionsProblem || (sessionsOk && !T.sessions.length ? "No captured phone sessions yet." : ""), onclick: () => { T.source = "session"; render(); } }, "Use a captured phone session")),
    T.source === "upload"
      ? h("div", { class: "field" }, h("label", { for: "teach-file" }, "Recording"), fileIn,
          h("span", { class: "hint" }, T.file ? `${T.file.name} · ${(T.file.size / 1048576).toFixed(1)} MB` : "Use iPhone Screen Recording, then pick the file here."))
      : missing || h("div", { class: "field" }, h("label", { for: "teach-session" }, "Captured session"), sessionSel),
    range,
    T.sessionsProblem ? h("p", { class: "hint" }, T.sessionsProblem) : null,
    h("div", { class: "field" }, h("label", { for: "teach-name" }, "Name"), name),
    h("div", { class: "field" }, h("label", { for: "teach-narr" }, "Describe the job"), narr),
    h("label", { class: "check small" }, ext, h("span", {}, "Let the AI model see this recording to read what happened. Without it, only on-device analysis runs.")),
    h("button", { class: "btn primary", type: "submit", disabled: !ready || T.learning }, T.learning ? "Learning…" : "Learn this workflow"));
  return form;
}

async function learn() {
  T.learning = true;
  render();
  try {
    let rec;
    if (T.source === "upload") {
      const m = await upload(T.file, () => {});
      rec = await post("/api/teach/recordings", { sha256: m.sha256, narration: T.narration.trim(), external_model_ok: T.externalOk });
    } else {
      rec = await post("/api/teach/sessions", { session_id: T.session, narration: T.narration.trim(),
        ...(T.first != null ? { first: T.first } : {}), ...(T.last != null ? { last: T.last } : {}) });
    }
    toast("Recording saved. Working out the steps…");
    T.routine = await post(`/api/teach/recordings/${rec.recording_id}/routine`, { name: T.name.trim() || "Taught workflow" });
    T.routines = null;
    T.file = null;
  } finally {
    T.learning = false;
    render();
  }
}

// The teaching view (steps + slots, questions, rules: tw-teach 78056ec4); a server without it still has the agency view.
async function openRoutine(id) {
  try {
    T.routine = await get(`/api/teach/routines/${encodeURIComponent(id)}`);
  } catch (err) {
    if (err.status !== 404) throw err;
    T.routine = await get(`/api/routines/${encodeURIComponent(id)}`);
  }
  T.inputs = {};
  render();
}

function review() {
  const r = T.routine;
  const rec = r.recording;
  const recId = r.recording_id || rec?.id;
  const steps = r.steps || [];
  const slots = Object.entries(r.slots || {});
  const frameUrl = (f) => (recId ? `/api/teach/recordings/${recId}/frames/${encodeURIComponent(f)}` : null);
  const answer = (q) => {
    const say = (text) => (e) => busy(e.currentTarget, async () => {
      if (!String(text()).trim()) return;
      await post(`/api/routines/${r.id}/corrections`, { text: `${q.id}: ${String(text()).trim()}` });
      await openRoutine(r.id);
    });
    const ta = h("input", { type: "text", placeholder: "Or say it in your words" });
    return h("div", { class: "card" },
      h("span", {}, q.ask || q.text || q.question || "", q.blocking === false ? h("span", { class: "tag", style: { marginLeft: "6px" } }, "Optional") : null),
      q.for === "publish" ? h("span", { class: "small muted" }, "Only matters when this workflow publishes.") : null,
      (q.options || []).length ? h("div", { class: "row-actions" }, q.options.map((o) => h("button", { class: "btn sm", type: "button", onclick: say(() => o) }, o))) : null,
      h("div", { class: "row-actions" }, ta, h("button", { class: "btn sm", type: "button", onclick: say(() => ta.value) }, "Answer")));
  };
  const corr = h("textarea", { rows: 2, placeholder: "e.g. Always open the profile from search, never from a link." });
  return h("div", { class: "teach-review" },
    h("button", { class: "btn quiet sm", onclick: () => { T.routine = null; openTeach(); } }, "← All workflows"),
    h("h2", { style: { marginTop: "8px" } }, r.name),
    h("div", { class: "small muted meta" },
      h("span", { class: `state ${r.confirmed_at ? "done" : "needs"}` }, r.confirmed_at ? "Confirmed" : "Check, then confirm"),
      r.version ? h("span", {}, `version ${r.version}`) : null,
      rec ? h("span", {}, `from ${rec.label === "fixture" ? "a fixture recording" : rec.name || "a recording"}${rec.duration_s ? ` · ${Math.round(rec.duration_s)} s` : ""}`) : null,
      r.interpreter ? h("span", { title: JSON.stringify(r.interpreter) }, "Details") : null),
    r.summary ? h("p", { class: "teach-summary" }, r.summary) : null,
    (r.questions || []).length ? h("div", { class: "answer" }, h("div", { class: "label" }, `It needs ${plural(r.questions.length, "answer")}`), r.questions.map(answer)) : null,
    slots.length ? h("div", { class: "answer" }, h("div", { class: "label" }, "Changes each time"),
      h("dl", { class: "kv" }, slots.map(([k, v]) => [h("dt", {}, k), h("dd", {}, `${v.type || "text"}${v.example ? ` · e.g. ${typeof v.example === "string" ? v.example : JSON.stringify(v.example)}` : ""}`)]))) : null,
    steps.length ? h("div", { class: "answer" }, h("div", { class: "label" }, `Steps · ${steps.length}`),
      h("ol", { class: "steps" }, steps.map((s, i) => {
        const ev = evidence(s);
        const unsure = s.uncertain || s.question || (s.confidence != null && s.confidence < 0.6);
        return h("li", { class: "step" },
          h("span", { class: "at" }, s.at != null ? `${Math.floor(s.at / 60)}:${String(Math.round(s.at % 60)).padStart(2, "0")}` : String(i + 1)),
          h("span", {}),
          h("span", { class: "txt" }, stepText(s),
            s.param ? h("span", { class: "tag", style: { marginLeft: "6px" } }, `changes: ${s.param}`) : null,
            s.evidence ? h("div", { class: "pace" }, s.evidence) : null,
            unsure ? h("div", { class: "pace", style: { color: "var(--warn)" } }, s.question || "Not sure about this step") : null),
          ev && recId ? h("button", { class: "thumb", "aria-label": `Show the evidence for step ${i + 1}`,
            onclick: () => ctx.showFrame(frameUrl(ev.frame), `Taught step ${i + 1} · from the recording`, ev.point) },
            h("img", { src: frameUrl(ev.frame), alt: "", loading: "lazy" })) : h("span", {}));
      }))) : h("p", { class: "muted small" }, "No steps were derived."),
    (r.rules || []).length ? h("div", { class: "answer" }, h("div", { class: "label" }, "Rules from your description"),
      h("ul", { class: "small", style: { margin: 0, paddingLeft: "1.1em" } }, r.rules.map((x) => h("li", {}, x.text || x.rule || String(x), x.quote ? h("span", { class: "muted" }, ` (“${x.quote}”)`) : null)))) : null,
    !r.confirmed_at ? h("div", { class: "row-actions" },
      h("button", { class: "btn primary", onclick: (e) => busy(e.currentTarget, async () => { await post(`/api/routines/${r.id}/confirm`, {}); await openRoutine(r.id); toast("Confirmed."); }) }, "Confirm this workflow")) : null,
    h("form", { class: "answer", onsubmit: (e) => { e.preventDefault(); busy(e.currentTarget.querySelector("button"), async () => {
      if (!corr.value.trim()) return corr.focus();
      await post(`/api/routines/${r.id}/corrections`, { text: corr.value.trim() });
      await openRoutine(r.id);
      toast("Saved as a correction; the next run uses it.");
    }); } }, h("div", { class: "label" }, "Correct something"), corr, h("button", { class: "btn sm", type: "submit" }, "Save correction")),
    r.confirmed_at ? runForm(r, slots) : null);
}

function runAvailability() {
  const kinds = ctx.kinds();
  const k = kinds?.find((x) => x.kind === "taught_run");
  const d = ctx.device();
  const why = !kinds ? "Checking what this phone can do…" : !k ? "This server has no taught-workflow runner."
    : !k.available ? ctx.plain(k) : !d || d.connection?.state !== "connected" ? d?.connection?.reason || "No phone connected." : "";
  return { why, key: `${!!kinds}|${k?.available}|${k?.reason}|${d?.connection?.state}` };
}

// Kinds and the device arrive after Teach may already be open: redraw just the run form when what it depends on
// changes, keeping the open workflow and whatever was typed (T.inputs).
let runKey = "";
export function teachStateChanged() {
  const el = ctx?.box.querySelector(".teach-run");
  if (!el || !T.routine?.confirmed_at || runAvailability().key === runKey) return;
  if (el.contains(document.activeElement)) return; // never under the person's cursor; the next update redraws it
  el.replaceWith(runForm(T.routine, Object.entries(T.routine.slots || {})));
}

function runForm(r, slots) {
  const { why, key } = runAvailability();
  runKey = key;
  // taught_run is read-only (a publishing routine goes through approvals in Jobs), so every slot is a typed value
  const media = slots.filter(([n, v]) => ["video", "image", "media"].includes(v.type) || n === "media");
  const fields = slots.filter((x) => !media.includes(x)).map(([name, v]) => {
    const id = `run-${name}`;
    return h("div", { class: "field" }, h("label", { for: id }, name.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase())),
      h("input", { type: "text", id, value: T.inputs[name] || "", placeholder: v.type === "handle" ? "@handle" : v.example ? `e.g. ${v.example}` : "",
        oninput: (e) => (T.inputs[name] = e.target.value) }));
  });
  // a workflow that posts media can't run read-only here at all: say where it runs, offer nothing that can't work
  if (media.length) return h("div", { class: "answer teach-run" }, h("div", { class: "label" }, "Run it"),
    h("p", { class: "small" }, "This workflow posts media, so it runs from Jobs and approvals, where each post is approved first. ",
      h("a", { href: "/jobs/#/new" }, "Open Jobs")));
  return h("form", { class: "answer teach-run", onsubmit: (e) => { e.preventDefault(); busy(e.currentTarget.querySelector("button[type=submit]"), async () => {
    const params = Object.fromEntries(Object.entries(T.inputs).map(([k, v]) => [k, String(v).trim().replace(/^@/, "")]).filter(([, v]) => v));
    await ctx.runTask("taught_run", { routine_id: r.id, params });
  }); } },

  h("div", { class: "label" }, "Run it on the phone"),
  fields.length ? fields : h("p", { class: "muted small" }, "This workflow takes no inputs."),
  h("button", { class: "btn primary", type: "submit", disabled: !!why, title: why }, why ? "Can't run yet" : "Run on the phone"),
  why ? h("p", { class: "hint" }, why) : null);
}
