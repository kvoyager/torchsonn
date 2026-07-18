import logging
import time

import pytest

from torchsonn.utils import ParamNamespace, abbrev_floats, timed_block


class TestParamNamespace:
    def test_get_set(self):
        ns = ParamNamespace()
        assert ns.get("missing") is None
        assert ns.get("missing", 42) == 42
        ns.set("a", 1)
        assert ns.get("a") == 1

    def test_getitem_setitem(self):
        ns = ParamNamespace()
        ns["k"] = "v"
        assert ns["k"] == "v"

    def test_pop_existing_and_missing(self):
        ns = ParamNamespace()
        ns.set("x", 5)
        assert ns.pop("x") == 5
        # popping again returns None (per current implementation)
        assert ns.pop("x") is None
        assert ns.pop("never-set") is None

    def test_dict_property(self):
        ns = ParamNamespace(a=1, b=2)
        assert ns.dict == {"a": 1, "b": 2}

    def test_from_dict_nested(self):
        ns = ParamNamespace.from_dict({"top": 1, "inner": {"a": 10}})
        assert ns.top == 1
        assert isinstance(ns.inner, ParamNamespace)
        assert ns.inner.a == 10


class TestTimedBlock:
    def test_runs_yields(self, caplog):
        caplog.set_level(logging.INFO)
        with timed_block("phase-1"):
            pass
        # at least one record mentioning phase-1 should be present
        assert any("phase-1" in rec.message for rec in caplog.records)

    def test_verbose_false_skips_log(self, caplog):
        caplog.set_level(logging.INFO)
        with timed_block("silent", verbose=False):
            pass
        assert not any("silent" in rec.message for rec in caplog.records)

    def test_no_name_still_logs(self, caplog):
        caplog.set_level(logging.INFO)
        with timed_block():
            time.sleep(0.001)
        assert any("Executed" in rec.message for rec in caplog.records)


class TestAbbrevFloats:
    def test_short_list_shown_inline(self):
        s = abbrev_floats([1.0, 2.0, 3.0])
        assert s == "[1.000, 2.000, 3.000]"

    def test_long_list_truncated(self):
        s = abbrev_floats(list(range(20)), edgeitems=2, threshold=5)
        assert s.startswith("[0.000, 1.000, ...")
        assert s.endswith("18.000, 19.000]")

    def test_threshold_boundary(self):
        # exactly threshold items → printed in full
        s = abbrev_floats([0.0] * 8, threshold=8)
        assert "..." not in s

    def test_custom_format(self):
        s = abbrev_floats([1.2345], fmt="{:.1f}")
        assert s == "[1.2]"
