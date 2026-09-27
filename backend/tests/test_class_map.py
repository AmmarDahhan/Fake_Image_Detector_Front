"""Tests for the provisional class -> binary verdict mapping.

The mapping is a single constant (``PROVISIONAL_BINARY_VERDICT_MAP``). These
tests exist to make the *provisional* status impossible to forget: if the AI
team changes a class name or the collapse changes, the reference
implementation and the expectations here must move together.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.schemas.analysis import Verdict
from app.services.model_class_map import (
    KNOWN_MODEL_CLASSES,
    PROVISIONAL_BINARY_VERDICT_MAP,
    verdict_for_class,
)


def _code_string_literals(path: Path) -> set[str]:
    """Every string literal in a module that is not a docstring.

    Docstrings are excluded because they are prose: a module may legitimately
    explain that the model outputs ``real`` or ``tampered`` without encoding
    either as a value.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))

    docstring_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_nodes.add(id(body[0].value))

    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstring_nodes
    }


class TestProvisionalMapping:
    def test_full_synthetic_maps_to_fake(self) -> None:
        assert verdict_for_class("full_synthetic") is Verdict.FAKE

    def test_real_maps_to_real(self) -> None:
        assert verdict_for_class("real") is Verdict.REAL

    def test_tampered_maps_to_fake(self) -> None:
        assert verdict_for_class("tampered") is Verdict.FAKE

    def test_mapping_has_exactly_three_entries(self) -> None:
        """A new class name must be a deliberate edit, not a silent addition."""
        assert len(PROVISIONAL_BINARY_VERDICT_MAP) == 3

    def test_every_value_is_a_valid_verdict(self) -> None:
        assert set(PROVISIONAL_BINARY_VERDICT_MAP.values()) <= {
            Verdict.REAL,
            Verdict.FAKE,
        }

    def test_both_verdicts_are_reachable(self) -> None:
        """Guards against a mapping that collapses everything to one verdict."""
        assert {v.value for v in PROVISIONAL_BINARY_VERDICT_MAP.values()} == {
            "real",
            "fake",
        }

    def test_known_classes_mirror_the_mapping_keys(self) -> None:
        assert KNOWN_MODEL_CLASSES == frozenset(PROVISIONAL_BINARY_VERDICT_MAP)

    def test_inference_plumbing_does_not_repeat_the_class_names(self) -> None:
        """The class order must be read from the checkpoint, not restated.

        ``real`` is also a member of the public ``Verdict`` enum, so a plain
        substring search is meaningless. This walks the AST instead and
        compares actual string literals, ignoring docstrings - so it fails if
        a class name is added as *code* anywhere in the plumbing, while
        tolerating prose that merely mentions the words.
        """
        import app

        app_dir = Path(app.__file__).parent
        # model_class_map.py is the single home for the mapping; the routes and
        # the service must not restate it.
        targets = [
            app_dir / "services" / "model_service.py",
            app_dir / "api" / "routes.py",
        ]

        for path in targets:
            found = _code_string_literals(path) & set(PROVISIONAL_BINARY_VERDICT_MAP)
            assert not found, (
                f"{path.name} contains the model class literal(s) "
                f"{sorted(found)}. The class order must come from the "
                "checkpoint; the only place the verdict mapping may appear is "
                "model_class_map.py."
            )


class TestUnmappedClass:
    def test_unknown_class_raises_rather_than_defaulting(self) -> None:
        """Defaulting an unknown class would answer with a verdict nobody chose."""
        with pytest.raises(KeyError):
            verdict_for_class("some_future_class")

    def test_class_lookup_is_case_sensitive(self) -> None:
        """A case difference is a real signal, not something to paper over."""
        with pytest.raises(KeyError):
            verdict_for_class("Real")

    def test_covered_classes_never_raise(self) -> None:
        for class_name in KNOWN_MODEL_CLASSES:
            assert verdict_for_class(class_name) in {Verdict.REAL, Verdict.FAKE}
