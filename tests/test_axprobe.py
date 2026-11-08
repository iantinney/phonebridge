"""The accessibility-tree probe against a fake audit daemon speaking the iOS 26.1 envelopes (axprobe.py)."""
import asyncio

import pytest

from phonebridge import axprobe


def env(kind, fields):
    return {"ObjectType": kind, "Value": {"ObjectType": "passthrough", "Value": fields}}


def pt(v):
    return {"ObjectType": "passthrough", "Value": v}


def elem(tok):
    return env("AXAuditElement_v1", {"PlatformElementValue_v1": pt(tok)})


def node(tok, desc, kids=()):
    return env("AXAuditNode_v1", {"AuditElementValue_v1": elem(tok), "HumanReadableDescriptionValue_v1": pt(desc),
                                  "HumanReadableRoleDescriptionValue_v1": pt("button"), "ChildrenValue_v1": pt(list(kids))})


TREE = {b"root": [b"a", b"b"], b"a": [b"a1"], b"b": [], b"a1": []}
DESC = {b"a": "Like, Button", b"b": "Comment, Button", b"a1": "Liked by many"}


class FakeAudit:
    def __init__(self, hierarchy_name="_AXHierarchyElementsAttribute"):
        self.calls, self.hname = [], hierarchy_name

    async def _invoke(self, selector, *args, expects_reply=True):
        self.calls.append(selector)
        if selector == "deviceCapabilities":
            return ["deviceFetchSpecialElement:", "deviceElement:valueForAttribute:"]
        if selector == "deviceApiVersion":
            return 7
        if selector == "deviceFetchSpecialElement:":
            return [elem(b"root")]
        if selector == "deviceElement:valueForAttribute:":
            tok = args[0]["Value"]["Value"]["PlatformElementValue_v1"]["Value"]
            name = args[1]["Value"]["Value"]["AttributeNameValue_v1"]["Value"]
            if name == "Label":
                return [pt(DESC.get(tok, ""))]
            if name != self.hname:
                return []
            return [node(tok, DESC.get(tok, "root"), [node(k, DESC[k]) for k in TREE[tok]])]
        if selector == "deviceFetchElementAtNormalizedDeviceCoordinate:":
            assert isinstance(args[0], axprobe.NSPointValue)
            return [elem(b"a")] if args[0].y == 0.15 else []
        raise AssertionError(selector)


def test_tree_walks_breadth_first_from_the_root_without_moving_focus():
    a = FakeAudit()
    out = asyncio.run(axprobe.AxProbe(a).run(["caps", "tree"]))
    assert out["caps"]["api_version"] == 7 and "deviceElement:valueForAttribute:" in out["caps"]["capabilities"]
    t = out["tree"]
    assert t["hierarchy_attr"] == "_AXHierarchyElementsAttribute"
    assert [(n["depth"], n["desc"]) for n in t["nodes"]] == [(1, "Like, Button"), (1, "Comment, Button"), (2, "Liked by many")]
    assert "deviceInspectorMoveWithOptions:" not in a.calls  # no focus move
    assert out["call_ms"]["hierarchy"] and out["op_ms"]["tree"] >= 0


def test_tree_tries_the_next_hierarchy_name_and_caps_nodes():
    a = FakeAudit(hierarchy_name="AXHierarchyElements")
    t = asyncio.run(axprobe.AxProbe(a).tree(max_nodes=2))
    assert t["hierarchy_attr"] == "AXHierarchyElements" and len(t["nodes"]) == 2


def test_hit_labels_a_returned_element_and_records_an_empty_reply():
    out = asyncio.run(axprobe.AxProbe(FakeAudit()).hit())
    assert [o["label"] for o in out] == [None, ["Like, Button"], None]


def test_a_failing_op_is_a_finding_and_an_unknown_op_is_an_error():
    class Broken(FakeAudit):
        async def _invoke(self, selector, *args, expects_reply=True):
            if selector == "deviceCapabilities":
                raise ConnectionError("gone")
            return await super()._invoke(selector, *args, expects_reply=expects_reply)
    out = asyncio.run(axprobe.AxProbe(Broken()).run(["caps", "tree"]))
    assert out["caps"]["error"].startswith("ConnectionError") and out["tree"]["nodes"]
    with pytest.raises(ValueError, match="caps \\| focus_first"):
        asyncio.run(axprobe.AxProbe(FakeAudit()).run(["nope"]))


def test_the_point_goes_out_as_an_nsvalue():
    from bpylist2 import archiver
    b = archiver.archive(axprobe.NSPointValue(0.5, 0.25))
    assert b"NSValue" in b and b"{0.5, 0.25}" in b
