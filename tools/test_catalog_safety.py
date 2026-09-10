#!/usr/bin/env python3
"""Guard: a rebuild must not quietly shrink the catalog.

The failure this prevents is specific. `build_catalog.py` reads every model from
HuggingFace; when that network is unwell, entries fail and would simply be
absent from the file it writes. Nobody notices, because a smaller catalog looks
exactly like a correct one. llmfit records a scrape that dropped architecture
metadata for 1,764 models in a single run, which is what these limits exist to
make impossible.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_catalog import (  # noqa: E402
    BUILD_LOSS_LIMIT,
    RESCUE_LIMIT,
    assess_loss,
    rescue_failures,
)


def entry(model_id, builds=1):
    return {
        "id": model_id,
        "display_name": model_id,
        "family": "Test",
        "architecture": {},
        "builds": [{"quant": f"Q{n}_K_M"} for n in range(builds)],
    }


def test_a_failed_model_is_carried_forward_not_dropped():
    previous = {"a": entry("a"), "b": entry("b")}
    models = [entry("a")]
    rescued = rescue_failures(models, previous, ["b"])
    assert rescued == ["b"]
    assert {m["id"] for m in models} == {"a", "b"}
    assert assess_loss(models, previous, rescued) == []


def test_a_model_with_no_history_cannot_be_rescued_and_is_reported():
    previous = {"a": entry("a")}
    models = [entry("a")]
    # "new" failed on its first ever build: nothing to carry forward, and
    # nothing was lost either.
    rescued = rescue_failures(models, previous, ["new"])
    assert rescued == []
    assert assess_loss(models, previous, rescued) == []


def test_a_rescue_is_never_a_duplicate():
    previous = {"a": entry("a")}
    models = [entry("a")]
    rescue_failures(models, previous, ["a"])
    assert len(models) == 1, "a model rebuilt and also rescued would appear twice"


def test_mass_failure_refuses_to_write():
    previous = {f"m{n}": entry(f"m{n}") for n in range(20)}
    models = [entry("m0")]
    failed = [f"m{n}" for n in range(1, 20)]
    rescued = rescue_failures(models, failed and previous, failed)
    reasons = assess_loss(models, previous, rescued)
    assert len(rescued) > RESCUE_LIMIT
    assert any("carried forward" in r for r in reasons), reasons


def test_a_sharp_drop_in_builds_refuses_to_write():
    previous = {"a": entry("a", builds=100)}
    models = [entry("a", builds=10)]
    reasons = assess_loss(models, previous, [])
    assert any("measured builds fell" in r for r in reasons), reasons

    # A small drop is a publisher tidying up, not an outage.
    barely = [entry("a", builds=int(100 * (1.0 - BUILD_LOSS_LIMIT / 2)))]
    assert assess_loss(barely, previous, []) == []


def test_a_vanished_model_refuses_to_write():
    previous = {"a": entry("a"), "gone": entry("gone")}
    models = [entry("a")]
    reasons = assess_loss(models, previous, [])
    assert any("disappeared entirely" in r for r in reasons), reasons


def test_the_first_ever_build_is_never_refused():
    models = [entry("a")]
    assert assess_loss(models, {}, []) == []


def main():
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    main()
