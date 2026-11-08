"""PER-05: the slow model for unfamiliar screens, on Ian's Claude plan (DECISIONS "Models via his plans", 04:3xZ 10-19:
no Anthropic API key). The director's warm `claude` process (`content_vlm.WarmClaude`: Haiku, Sonnet on escalation)
answers in a closed JSON schema, images at ≤ 585×1266; AWS Bedrock only when the plan reports a usage limit. A per-day
spend cap (default $2, metered spend only: Bedrock) with an alert at 80 %; every call is logged (frame hash, prompt
hash, answer, usage, cost, provider) and cached by frame hash + prompt hash so replays never call out.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from datetime import date
from hashlib import sha256
from pathlib import Path

from PIL import Image

from ..config import DATA_DIR, site

MODEL = "claude-haiku-5-5"
PRICES = {  # USD per million tokens (input, output); Haiku 5.5 up to 100K-token prompts
    "claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}
MAX_IMAGE = (585, 1266)
STATE = Path(os.environ.get("SCREENS_STATE", DATA_DIR / "screens")).expanduser()

RESOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "screen_name": {"type": "string", "description": "short snake_case name for this screen or sheet"},
        "kind": {"type": "string", "enum": ["screen", "popup", "sheet", "alert", "captcha", "two_factor", "payment",
                                            "warning", "action_block", "login", "settings_toggle", "other"]},
        "action": {"type": "string", "enum": ["dismiss", "back", "scroll", "tap_named", "open", "none"]},
        "element_id": {"type": "integer", "description": "the [n] of the listed element to tap, or -1 when no listed element fits"},
        "target_box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                       "description": "[x0, y0, x1, y1] as fractions of the screen for the control to tap (only when element_id is -1)"},
        "label_text": {"type": "string", "description": "the exact visible label of the control to tap, or empty"},
        "confidence": {"type": "number", "description": "0..1: how sure you are of the screen name and the action"},
        "reason": {"type": "string"},
    },
    "required": ["screen_name", "kind", "action", "element_id", "target_box", "label_text", "confidence", "reason"],
    "additionalProperties": False,
}

SYSTEM = """You look at one iPhone screenshot from a social app and decide the single safest action to get the app
back onto a known screen. Name the screen by its UI chrome (bars, buttons, sheets, dialogs), never by the
content (a video or photo filling the screen is just the feed). Allowed actions: dismiss a sheet or popup
with its non-committal button (Not now, Cancel, Close, Skip, Maybe later, Don't allow, OK on an info
dialog), go back (back chevron or X), scroll, tap a named navigation control (a tab, a back button), open a
sheet, or none. Never choose Post, Share, Send, Delete, Follow, Unfollow, Buy, Log in, Sign up, Allow, a
Settings toggle, or a text field; never type. If the screen is a captcha, a two-factor prompt, a payment
sheet, a warning or an action block, say so in `kind` and choose action none. Answer in the JSON schema."""


class CapReached(RuntimeError):
    pass


class VLMError(RuntimeError):
    pass


def shrink(png: bytes, max_size=MAX_IMAGE) -> bytes:
    im = Image.open(io.BytesIO(png))
    if im.width <= max_size[0] and im.height <= max_size[1]:
        return png
    im.thumbnail(max_size, Image.LANCZOS)
    out = io.BytesIO()
    im.convert("RGB").save(out, format="PNG", optimize=True)
    return out.getvalue()


CLI_MODELS = {"claude-haiku-5-5": "haiku", "claude-sonnet-5-5": "sonnet", "claude-opus-5-5": "opus"}
# AWS Bedrock through a configured AWS profile (else the default credential chain), 10 requests/min per model. The
# logical models map to the ones a Bedrock account offers.
BEDROCK_REGION = os.environ.get("SCREENS_BEDROCK_REGION", "us-east-1")
BEDROCK_PROFILE = os.environ.get("SCREENS_BEDROCK_PROFILE", site("BEDROCK_PROFILE", None))
BEDROCK_MODELS = {"claude-haiku-5-5": "us.anthropic.claude-haiku-4-5-20251001-v1:0", "claude-sonnet-5-5": "us.anthropic.claude-sonnet-4-6",
                  "claude-opus-5-5": "us.anthropic.claude-sonnet-4-6"}
BEDROCK_PRICES = {"us.anthropic.claude-haiku-4-5-20251001-v1:0": (1.0, 5.0), "us.anthropic.claude-sonnet-4-6": (3.0, 15.0)}  # USD/MTok, estimate
BEDROCK_PER_MIN = 10
_bedrock_calls: dict[str, list[float]] = {}


def bedrock_budget(model: str, now: float | None = None, per_min: int = BEDROCK_PER_MIN) -> float:
    """Seconds to wait before another call to `model` fits the per-minute limit (0 when it fits now); records the call."""
    now = time.time() if now is None else now
    calls = [t for t in _bedrock_calls.get(model, []) if now - t < 60]
    wait = 0.0 if len(calls) < per_min else 60 - (now - calls[0]) + 0.05
    calls.append(now + wait)
    _bedrock_calls[model] = calls
    return wait


BEDROCK_JPEG_QUALITY = 80


def jpeg_block(block: dict, quality: int = BEDROCK_JPEG_QUALITY, max_size=MAX_IMAGE) -> dict:
    """A base64 image block re-encoded as JPEG at `quality`, no wider than `max_size` (controller-verify's bake-off:
    the smaller JPEG saves ~25 % of Bedrock's latency against the PNG, same answers)."""
    src = block.get("source") or {}
    if block.get("type") != "image" or src.get("type") != "base64":
        return block
    im = Image.open(io.BytesIO(base64.b64decode(src["data"])))
    if im.width > max_size[0] or im.height > max_size[1]:
        im.thumbnail(max_size, Image.LANCZOS)
    out = io.BytesIO()
    im.convert("RGB").save(out, format="JPEG", quality=quality, optimize=True)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(out.getvalue()).decode()}}


def bedrock_payload(payload: dict) -> dict:
    """The Messages payload for Bedrock's models: 4.x takes no `effort` and ignores output_config, so the JSON schema
    goes into the prompt as an instruction (a forced tool answered the same but cost +49 % latency in
    controller-verify's bake-off) and images go as JPEG q80 at ≤ 585 px."""
    out = {k: v for k, v in payload.items() if k != "output_config"}
    out["model"] = BEDROCK_MODELS.get(payload["model"], payload["model"])
    schema = ((payload.get("output_config") or {}).get("format") or {}).get("schema")
    if schema:
        rule = ("Answer with one JSON object only, no prose and no code fence, valid against this JSON schema: "
                + json.dumps(schema, separators=(",", ":")))
        out["system"] = (payload.get("system") + "\n\n" + rule) if payload.get("system") else rule
    out["messages"] = [{**m, "content": [jpeg_block(b) if isinstance(b, dict) else b for b in m["content"]]}
                       if isinstance(m.get("content"), list) else m for m in payload.get("messages", [])]
    return out


_bedrock_clients: dict = {}


def _bedrock_client(anthropic, timeout: float):
    """One client per process: building it resolves the AWS profile (seconds), the calls themselves are ~1.5 s."""
    if timeout not in _bedrock_clients:
        _bedrock_clients[timeout] = anthropic.AnthropicBedrock(aws_region=BEDROCK_REGION, aws_profile=BEDROCK_PROFILE, timeout=timeout, max_retries=1)
    return _bedrock_clients[timeout]


def bedrock_transport(payload: dict, timeout: float = 60.0) -> dict:
    import anthropic
    body = bedrock_payload(payload)
    if (wait := bedrock_budget(body["model"])) > 0:
        time.sleep(wait)
    client = _bedrock_client(anthropic, timeout)
    try:  # the schema rides in the system prompt (bedrock_payload); the answer is a text block that extract_json reads
        resp = client.messages.create(model=body["model"], max_tokens=body["max_tokens"], system=body.get("system") or anthropic.NOT_GIVEN,
                                      messages=body["messages"])
    except anthropic.RateLimitError as e:
        raise VLMError(f"bedrock rate limited: {e}") from None
    except anthropic.APIStatusError as e:
        raise VLMError(f"bedrock HTTP {e.status_code}: {str(e)[:300]}") from None
    except (anthropic.APIConnectionError, Exception) as e:
        raise VLMError(f"bedrock unreachable: {type(e).__name__}: {str(e)[:200]}") from None
    d = resp.model_dump()
    tool = next((b for b in d.get("content", []) if b.get("type") == "tool_use"), None)
    if tool is not None:  # a model that still answers through a tool: hand the common parser its input as JSON
        d["content"] = [{"type": "text", "text": json.dumps(tool.get("input") or {})}]
    pi, po = BEDROCK_PRICES.get(body["model"], (1.0, 5.0))
    u = d.get("usage", {})
    d["cost_usd"] = u.get("input_tokens", 0) * pi / 1e6 + u.get("output_tokens", 0) * po / 1e6
    d["provider"] = "bedrock"
    return d


def extract_json(text: str) -> dict:
    """The first JSON object in a text answer (a model without structured output still answers in JSON)."""
    try:
        return json.loads(text)
    except ValueError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    raise VLMError(f"answer is not JSON: {text[:200]!r}")


# the CLI's own usage-limit notices ("You've hit your limit · resets 3pm", "Claude AI usage limit reached", a 429), the
# same words lanewatch watches for: the one case that falls back to Bedrock
PLAN_LIMIT = re.compile(r"hit your (?:\w+ )?limit|usage limit|limit reached|out of (?:extra )?usage|rate.?limit|\b429\b", re.I)
_warm_pool: dict = {}  # (cli model, system, schema) -> one warm claude process, shared by every VLM in this process


NOT_HERE = re.compile(r"not logged in|/login|no such file|not found", re.I)  # the CLI can't answer on this host (vs a plan limit)
_remote = {"on": False}


def claude_via() -> list[str] | None:
    """Where the plan's CLI runs when it can't here: $PHONEBRIDGE_CLAUDE_VIA (e.g. "ssh -o BatchMode=yes myhost", a host
    whose claude is logged in), else the deployment's route; none by default. "" turns it off."""
    via = os.environ.get("PHONEBRIDGE_CLAUDE_VIA", site("CLAUDE_VIA", ""))
    return via.split() if via.strip() else None


def plan_transport(payload: dict, timeout: float = 120.0) -> dict:
    """The Messages-shaped payload through the director's warm `claude` process on the CLI's plan: one process per model,
    system prompt and schema, kept warm (≈ 3 s per answer after a one-off startup). Returns a Messages-like response;
    the plan is not metered spend (cost 0). When the CLI is not logged in (or absent) on this host, the same process runs
    on `claude_via()` (the Mac's daemon has no keychain access over its own launch; measured 10-20: 'Not logged in')."""
    from ..director.content_vlm import WarmClaude
    model = CLI_MODELS.get(payload["model"], payload["model"])
    schema = (payload.get("output_config") or {}).get("format", {}).get("schema")
    content = payload["messages"][0]["content"]
    png = base64.b64decode(next(b["source"]["data"] for b in content if b.get("type") == "image"))
    text = "\n".join(b["text"] for b in content if b.get("type") == "text")

    def ask(remote: bool) -> dict:
        key = (model, payload.get("system"), json.dumps(schema, sort_keys=True), remote)
        if key not in _warm_pool:
            _warm_pool[key] = WarmClaude(model, timeout=timeout, system=payload.get("system"), schema=schema,
                                         **({"via": claude_via()} if remote else {}))
        return _warm_pool[key]._ask(png, text)
    try:
        ans = ask(_remote["on"])
    except Exception as e:  # noqa: BLE001 - only "can't answer here" moves it; a plan limit or a bad answer propagates
        if _remote["on"] or not claude_via() or not NOT_HERE.search(f"{type(e).__name__}: {e}"):
            raise
        _remote["on"] = True
        ans = ask(True)
    return {"content": [{"type": "text", "text": json.dumps(ans)}], "stop_reason": "end_turn", "usage": {}, "cost_usd": 0.0,
            "provider": "claude-plan"}


def route() -> str:
    """Which model names unfamiliar screens, from $PHONEBRIDGE_VLM: "claude-cli" (the signed-in claude CLI here; AWS
    Bedrock on its usage limit, with the vlm extra), "bedrock", or "off", the default (no screenshot leaves the Mac)."""
    return os.environ.get("PHONEBRIDGE_VLM", site("VLM", "off"))


class VLM:
    """The model `route()` names: the local claude CLI's plan (the warm claude process), Bedrock on a plan usage limit;
    a `transport` callable replaces both (tests, a local model later)."""

    def __init__(self, model: str = MODEL, cap_usd: float = 2.0, state_dir: Path = STATE, notify=None, transport=None,
                 effort: str = "low", cache: bool = True, timeout: float = 120.0, escalate_model: str = "claude-sonnet-5-5"):
        self.model, self.cap, self.effort, self.timeout = model, cap_usd, effort, timeout
        self.escalate_model = escalate_model  # when Haiku is unsure or its action failed the next-screen check
        self.state = Path(state_dir)
        self.notify = notify
        self.transport = transport
        self.use_cache = cache
        self._alerted = False
        self.last_cost: float | None = None
        self.last_provider: str | None = None

    def _send(self, payload: dict) -> dict:
        if self.transport is not None:
            return self.transport(payload)
        if (r := route()) not in ("claude-cli", "bedrock"):
            raise VLMError("no model for unfamiliar screens: set PHONEBRIDGE_VLM=claude-cli or bedrock to use one")
        if r == "bedrock":
            return bedrock_transport(payload, self.timeout)
        try:
            return plan_transport(payload, self.timeout)
        except Exception as e:
            if not PLAN_LIMIT.search(str(e)):
                raise VLMError(f"claude plan: {type(e).__name__}: {str(e)[:300]}") from None
            self._log_note(f"Claude plan limit, falling back to Bedrock: {str(e)[:200]}")
        return bedrock_transport(payload, self.timeout)

    def _log_note(self, note: str) -> None:
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        with open(self.state / "vlm.jsonl", "a") as fh:
            fh.write(json.dumps({"ts": round(time.time(), 3), "note": note}) + "\n")

    # --- spend ----------------------------------------------------------------------------------
    @property
    def ledger(self) -> Path:
        return self.state / "vlm-spend.json"

    def spend_today(self) -> float:
        try:
            return float(json.loads(self.ledger.read_text()).get(str(date.today()), 0.0))
        except (OSError, ValueError):
            return 0.0

    def _add_spend(self, usd: float) -> float:
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            d = json.loads(self.ledger.read_text())
        except (OSError, ValueError):
            d = {}
        d[str(date.today())] = round(d.get(str(date.today()), 0.0) + usd, 8)
        self.ledger.write_text(json.dumps(d, indent=1))
        total = d[str(date.today())]
        if total >= 0.8 * self.cap and not self._alerted:
            self._alerted = True
            if self.notify:
                self.notify(f"VLM spend ${total:.4f} is over 80% of today's ${self.cap:.2f} cap")
        return total

    @staticmethod
    def cost(model: str, usage: dict) -> float:
        pi, po = PRICES.get(model, PRICES[MODEL])
        return usage.get("input_tokens", 0) * pi / 1e6 + usage.get("output_tokens", 0) * po / 1e6

    # --- calls ----------------------------------------------------------------------------------
    def ask(self, png: bytes, prompt: str, schema: dict, system: str = SYSTEM, max_tokens: int = 512,
            tag: str = "screens", fresh: bool = False) -> dict:
        """One structured call about one frame. Cached by frame hash + prompt hash (replays never call the
        network); `fresh` bypasses the cache (a retry after the cached answer failed); logged; capped."""
        frame = sha256(png).hexdigest()
        phash = sha256(json.dumps([system, prompt, schema, self.model, self.effort]).encode()).hexdigest()[:16]
        cache = self.state / "vlm-cache" / f"{frame[:16]}-{phash}.json"
        if self.use_cache and not fresh and cache.exists():
            ans = json.loads(cache.read_text())
            self._log(frame, phash, ans, {}, 0.0, 0.0, cached=True, tag=tag)
            return ans
        est = self.last_cost or self.cost(self.model, {"input_tokens": 600, "output_tokens": 60})
        if self.spend_today() + est > self.cap:
            raise CapReached(f"VLM day cap ${self.cap:.2f} reached (${self.spend_today():.4f} spent)")
        img = base64.b64encode(shrink(png)).decode()
        payload = {
            "model": self.model, "max_tokens": max_tokens, "system": system,
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
                {"type": "text", "text": prompt}]}],
        }
        t0 = time.perf_counter()
        resp = self._send(payload)
        ms = round((time.perf_counter() - t0) * 1000)
        self.last_provider = resp.get("provider") or ("transport" if self.transport is not None else None)
        if resp.get("stop_reason") == "refusal":
            raise VLMError(f"refused: {resp.get('stop_details')}")
        text = next((b.get("text") for b in resp.get("content", []) if b.get("type") == "text"), None)
        if text is None:
            raise VLMError(f"no text block in the answer: {resp}")
        ans = extract_json(text)
        usage = resp.get("usage", {})
        usd = resp["cost_usd"] if resp.get("cost_usd") is not None else self.cost(self.model, usage)
        self.last_cost = usd
        self._add_spend(usd)
        self._log(frame, phash, ans, usage, usd, ms, cached=False, tag=tag)
        if self.use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            cache.write_text(json.dumps(ans))
        return ans

    def _log(self, frame, phash, ans, usage, usd, ms, cached, tag) -> None:
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        with open(self.state / "vlm.jsonl", "a") as fh:
            fh.write(json.dumps({"ts": round(time.time(), 3), "tag": tag, "model": self.model, "provider": self.last_provider, "frame": frame,
                                 "prompt_hash": phash, "answer": ans, "usage": usage, "cost_usd": round(usd, 7),
                                 "ms": ms, "cached": cached}) + "\n")

    def resolve(self, png: bytes, ocr: list[dict], known: list[str], platform: str, context: str | None = None,
                width: int = 1170, height: int = 2532, escalate: bool = False, fresh: bool = False) -> dict:
        """SCR-06: name an unknown screen and propose one safe action. `escalate` uses the stronger model."""
        if escalate and self.escalate_model and self.model != self.escalate_model:
            strong = VLM(model=self.escalate_model, cap_usd=self.cap, state_dir=self.state, notify=self.notify,
                         transport=self.transport, effort=self.effort, cache=self.use_cache, timeout=self.timeout, escalate_model="")
            return strong.resolve(png, ocr, known, platform, context, width, height, escalate=False, fresh=True)
        elements = numbered_elements(ocr)
        lines = [f"[{n}] {i['text'].strip()!r} at ({i['x'] / width:.2f}, {i['y'] / height:.2f})" for n, i in elements]
        prompt = (f"App: {platform}. Known screens (do not reuse their names unless it is one of them): "
                  f"{', '.join(known[:80]) or 'none yet'}.\n"
                  + (f"Context: {context}\n" if context else "")
                  + "Numbered elements the OCR read, with positions as fractions of the screen (tap by element_id; "
                    "icons without text are not listed, use target_box for those):\n" + "\n".join(lines)
                  + "\nName this screen and choose the one safest action.")
        return self.ask(png, prompt, RESOLVE_SCHEMA, tag="resolve", fresh=fresh or escalate)


def numbered_elements(ocr: list[dict], min_conf: float = 0.5, limit: int = 60) -> list[tuple[int, dict]]:
    """Grounding by numbered elements, not coordinates: the OCR boxes (an icon/button detector joins later)."""
    return list(enumerate([i for i in ocr if i.get("conf", 1.0) >= min_conf][:limit], start=1))
