"""EXPERIMENT (accessibility-tree spike, orchestrator 2026-10-19): what Apple's accessibility audit daemon (the remote
server behind Xcode's Accessibility Inspector) gives us about the foreground app, and how fast, to compare with OCR
for grounding. Selectors and object keys are from an iOS 26.1 decompile and go-ios (research notes:
~/.local/state/agent-reach/evidence/accessibility/; pymobiledevice3 `AccessibilityAudit` is the transport).

Ops (each timed):
  caps         deviceCapabilities + deviceApiVersion: the live selector list.
  focus_first  move the inspector focus to the first element and keep the whole focus event, whose sections name the
               attributes this iOS serves (the Hierarchy attribute's string among them). MOVES FOCUS: the daemon
               scrolls the element into view, so only on a screen that may scroll (the probe page, the home screen).
  tree         from the app's root element (deviceFetchSpecialElement:, no focus move) one Hierarchy read per element,
               breadth first, up to `max_nodes`: role, description, class per node. iOS 26.1 has no frame attribute.
  hit          deviceFetchElementAtNormalizedDeviceCoordinate: at a few points, then the element's Label.
  audit        deviceBeginAuditTypes: [100, 1000, 5000]: the one place rects appear (flagged elements only).
Selection mode (monitored event type 2) is never armed: on iOS 27 the daemon then swallows touches."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from bpylist2 import archiver

HIERARCHY_GUESSES = ("_AXHierarchyElementsAttribute", "AXHierarchyElements", "Hierarchy")
HIT_POINTS = ((0.5, 0.5), (0.5, 0.15), (0.5, 0.9))
AUDIT_TYPES = [100, 1000, 5000]
CALL_TIMEOUT_S = 5.0


class NSPointValue:
    """NSValue(CGPoint) on the wire (NSKeyedArchiver: NS.special 1 = point, NS.pointval "{x, y}")."""

    def __init__(self, x: float, y: float):
        self.x, self.y = x, y

    def encode_archive(self, ao) -> None:
        ao.encode("NS.special", 1)
        ao.encode("NS.pointval", f"{{{self.x:g}, {self.y:g}}}")


archiver.ARCHIVE_CLASS_MAP[NSPointValue] = "NSValue"


def plain(d: Any) -> Any:
    """The daemon's {ObjectType, Value} envelopes as plain data; typed objects keep their type under "_type"."""
    if isinstance(d, list):
        return [plain(x) for x in d]
    if isinstance(d, (bytes, bytearray)):
        return bytes(d).hex()
    if not isinstance(d, dict):
        return d
    if "ObjectType" in d:
        v = plain(d.get("Value"))
        if d["ObjectType"] == "passthrough":
            return v
        return {"_type": d["ObjectType"], **v} if isinstance(v, dict) else {"_type": d["ObjectType"], "value": v}
    return {k: plain(v) for k, v in d.items()}


def element(token: bytes) -> dict:
    return {"ObjectType": "AXAuditElement_v1", "Value": {"ObjectType": "passthrough", "Value": {
        "PlatformElementValue_v1": {"ObjectType": "passthrough", "Value": token}}}}


def attribute(name: str) -> dict:
    return {"ObjectType": "AXAuditElementAttribute_v1", "Value": {"ObjectType": "passthrough", "Value": {
        "AttributeNameValue_v1": {"ObjectType": "passthrough", "Value": name}}}}


def token_of(raw: Any) -> bytes | None:
    """The opaque element token inside a raw (not yet `plain`) AXAuditElement_v1 or AXAuditNode_v1 reply."""
    if isinstance(raw, list):
        return next((t for t in (token_of(x) for x in raw) if t), None)
    if isinstance(raw, dict):
        if "PlatformElementValue_v1" in raw:
            v = raw["PlatformElementValue_v1"]
            v = v.get("Value") if isinstance(v, dict) else v
            return bytes(v) if isinstance(v, (bytes, bytearray)) else None
        for k in ("Value", "AuditElementValue_v1", "ElementValue_v1"):
            if k in raw and (t := token_of(raw[k])):
                return t
    return None


def children_of(raw: Any) -> list:
    """The raw child nodes of a Hierarchy reply (one AXAuditNode_v1 whose ChildrenValue_v1 lists direct children)."""
    if isinstance(raw, list):
        return [c for x in raw for c in children_of(x)]
    if isinstance(raw, dict):
        if "ChildrenValue_v1" in raw:
            c = raw["ChildrenValue_v1"]
            c = c.get("Value") if isinstance(c, dict) and "ObjectType" in c else c
            return c if isinstance(c, list) else []
        if "Value" in raw:
            return children_of(raw["Value"])
    return []


def node_summary(raw: Any) -> dict:
    p = plain(raw)
    p = p[0] if isinstance(p, list) and p else p
    if not isinstance(p, dict):
        return {"value": p}
    return {k: p.get(f) for k, f in (("role", "HumanReadableRoleDescriptionValue_v1"),
                                     ("desc", "HumanReadableDescriptionValue_v1"), ("cls", "ClassNameValue_v1"),
                                     ("ignored", "IsIgnoredValue_v1")) if p.get(f) is not None}


class AxProbe:
    def __init__(self, audit):
        self.audit = audit  # pymobiledevice3 AccessibilityAudit, entered
        self.timings: dict[str, list[float]] = {}

    async def call(self, op: str, selector: str, *args, reply: bool = True):
        t = time.perf_counter()
        try:
            return await asyncio.wait_for(self.audit._invoke(selector, *args, expects_reply=reply), CALL_TIMEOUT_S)
        finally:
            self.timings.setdefault(op, []).append(round((time.perf_counter() - t) * 1000, 1))

    async def caps(self) -> dict:
        caps = await self.call("caps", "deviceCapabilities")
        try:
            api = await self.call("caps", "deviceApiVersion")
        except Exception as e:  # noqa: BLE001 — older daemons lack it
            api = f"{type(e).__name__}"
        return {"capabilities": plain(caps), "api_version": plain(api)}

    async def focus_first(self) -> dict:
        from pymobiledevice3.services.accessibilityaudit import Direction
        await self.audit.set_show_visuals(False)
        await self.audit.set_app_monitoring_enabled(True)
        await self.audit.set_monitored_event_type(0)
        q = self.audit._event_queue
        t = time.perf_counter()
        await self.audit.move_focus(Direction.First)
        end = t + 3.0
        while (left := end - time.perf_counter()) > 0:
            try:
                name, args = await asyncio.wait_for(q.get(), left)
            except asyncio.TimeoutError:
                break
            if name == "hostInspectorCurrentElementChanged:":
                self.timings.setdefault("focus_first", []).append(round((time.perf_counter() - t) * 1000, 1))
                event = plain(self.audit._extract_event_payload(args))
                return {"event": event, "attribute_names": sorted(set(_attribute_names(event)))}
        return {"event": None, "error": "no focus event within 3 s"}

    async def tree(self, max_nodes: int = 120, hierarchy_attr: str | None = None) -> dict:
        root = await self.call("root", "deviceFetchSpecialElement:", 0)
        tok = token_of(root)
        if tok is None:
            return {"root": plain(root), "nodes": [], "error": "no root token"}
        names = (hierarchy_attr,) if hierarchy_attr else HIERARCHY_GUESSES
        used, nodes, queue = None, [], [(tok, 0)]
        for name in names:  # the first name that yields children is this iOS's Hierarchy attribute
            reply = await self.call("hierarchy", "deviceElement:valueForAttribute:", element(tok), attribute(name))
            if children_of(reply):
                used = name
                break
        if used is None:
            return {"root": plain(root), "nodes": [], "error": f"no children under {names}"}
        seen: set[bytes] = set()
        while queue and len(nodes) < max_nodes:
            tok, depth = queue.pop(0)
            if tok in seen:
                continue
            seen.add(tok)
            reply = await self.call("hierarchy", "deviceElement:valueForAttribute:", element(tok), attribute(used))
            for child in children_of(reply):
                ctok = token_of(child)
                nodes.append({"depth": depth + 1, **node_summary(child)})
                if ctok and ctok not in seen:
                    queue.append((ctok, depth + 1))
                if len(nodes) >= max_nodes:
                    break
        return {"hierarchy_attr": used, "nodes": nodes}

    async def hit(self, points=HIT_POINTS) -> list[dict]:
        out = []
        for x, y in points:
            reply = await self.call("hit", "deviceFetchElementAtNormalizedDeviceCoordinate:", NSPointValue(x, y))
            tok = token_of(reply)
            label = None
            if tok:
                label = plain(await self.call("label", "deviceElement:valueForAttribute:", element(tok), attribute("Label")))
            out.append({"point": [x, y], "reply": plain(reply) if tok is None else "token", "label": label})
            await asyncio.sleep(0.15)  # the daemon caches a hit for 0.1 s
        return out

    async def audit_rects(self, types=AUDIT_TYPES) -> dict:
        t = time.perf_counter()
        issues = await asyncio.wait_for(self.audit.run_audit(list(types)), 60)
        self.timings.setdefault("audit", []).append(round((time.perf_counter() - t) * 1000, 1))
        rows = [{"type": plain(i._fields.get("auditTestTypeValue_v1")), "rect": plain(i._fields.get("ElementRectValue_v1")),
                 "desc": plain(i._fields.get("ElementDescriptionValue_v1")),
                 "ml": plain(i._fields.get("MLGeneratedDescriptionValue_v1")),
                 "text": plain(i._fields.get("ElementTextValue_v1"))} for i in issues]
        return {"issues": rows}

    async def run(self, ops: list[str], max_nodes: int = 120) -> dict:
        out: dict[str, Any] = {}
        hierarchy_attr = None
        for op in ops:
            t = time.perf_counter()
            try:
                if op == "caps":
                    out[op] = await self.caps()
                elif op == "focus_first":
                    out[op] = await self.focus_first()
                    hierarchy_attr = next((n for n in out[op].get("attribute_names", []) if "ierarch" in n), None)
                elif op == "tree":
                    out[op] = await self.tree(max_nodes, hierarchy_attr)
                elif op == "hit":
                    out[op] = await self.hit()
                elif op == "audit":
                    out[op] = await self.audit_rects()
                else:
                    raise ValueError(f"ax_probe op {op!r}: caps | focus_first | tree | hit | audit")
            except ValueError:
                raise
            except Exception as e:  # noqa: BLE001 — a failing op is a finding; the rest still run
                out[op] = {"error": f"{type(e).__name__}: {e}"[:300]}
            out.setdefault("op_ms", {})[op] = round((time.perf_counter() - t) * 1000, 1)
        out["call_ms"] = self.timings
        return out


def _attribute_names(d: Any):
    if isinstance(d, dict):
        if "AttributeNameValue_v1" in d:
            yield str(d["AttributeNameValue_v1"])
        for v in d.values():
            yield from _attribute_names(v)
    elif isinstance(d, list):
        for v in d:
            yield from _attribute_names(v)
