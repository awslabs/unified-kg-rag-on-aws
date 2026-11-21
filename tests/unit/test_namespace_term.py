# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One user-facing name, ``namespace``, on every surface.

The namespace was the *index suffix* (``--suffix``, ``RAGInput.suffix``)
and the *index value* (``processing.document_parsing.index_value``, the
console's ingest request). Each surface takes ``namespace``; each old name
still works as a deprecated alias, with a warning, through the one mapping
of ``domain/models/namespace.py`` (and ``DEPRECATED_KEYS`` for the config
key); an old and the new name given two different values are refused.
"""

from __future__ import annotations

import argparse
import inspect
import re
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tests.unit.test_cli_conventions import ALL_PARSERS
from unified_kg_rag.application.config import ConfigLoader, env_var
from unified_kg_rag.application.retrieval.rag_chain import RAGInput
from unified_kg_rag.domain.models import Config, EvaluationQuery
from unified_kg_rag.domain.models.config import _root
from unified_kg_rag.domain.models.namespace import DEPRECATED_NAMESPACE_NAMES
from unified_kg_rag.shared.utils.deprecation import PackageDeprecationWarning

pytestmark = pytest.mark.unit

ROOT = Path(__file__).parents[2]
# The CLIs that read or write a namespace (the others touch no index).
NAMESPACE_CLIS = {
    "run-ingestion",
    "run-rag",
    "run-eval",
    "run-visualization",
    "run-describe",
    "run-mcp-server",
    "run-console",
}
# Each CLI's other required flags.
_REQUIRED = {"run-rag": ["--query", "q"], "run-eval": ["--data-path", "eval.json"]}


@pytest.fixture(autouse=True)
def _warn_afresh() -> Iterator[None]:
    # The config's WARNING is logged once per key and process.
    _root._warn_deprecated.cache_clear()
    yield
    _root._warn_deprecated.cache_clear()


def _parse(cli: str, *flags: str) -> argparse.Namespace:
    return ALL_PARSERS[cli]().parse_args([*_REQUIRED.get(cli, []), *flags])


def _flags(parser: argparse.ArgumentParser) -> set[str]:
    return {o for a in parser._actions for o in a.option_strings}


def test_the_clis_that_take_a_namespace_take_it_as_namespace() -> None:
    # A new CLI that names a namespace takes --namespace (and the alias).
    taking = {
        name for name, build in ALL_PARSERS.items() if "--namespace" in _flags(build())
    }
    assert taking == NAMESPACE_CLIS
    for name in NAMESPACE_CLIS:
        assert "--suffix" in _flags(ALL_PARSERS[name]()), name


@pytest.mark.parametrize("cli", sorted(NAMESPACE_CLIS))
def test_namespace_and_its_deprecated_alias_parse_alike(
    cli: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _parse(cli, "--namespace", "tenant-a").namespace == "tenant-a"
    assert "deprecated" not in capsys.readouterr().err
    assert _parse(cli, "--suffix", "tenant-a").namespace == "tenant-a"
    assert "--suffix is deprecated" in capsys.readouterr().err
    # Both naming one namespace is not a conflict.
    assert _parse(cli, "--suffix", "a", "--namespace", "a").namespace == "a"
    assert _parse(cli).namespace is None
    # The alias is left out of --help.
    shown = ALL_PARSERS[cli]().format_help()
    assert "--namespace" in shown
    assert not re.search(r"^\s+--suffix|\[--suffix", shown, re.MULTILINE)


@pytest.mark.parametrize("cli", sorted(NAMESPACE_CLIS))
@pytest.mark.parametrize(
    "flags",
    [
        ["--namespace", "a", "--suffix", "b"],
        ["--suffix", "a", "--namespace", "b"],
        ["--namespace", "a", "--namespace", "b"],
        ["--namespace", "Bad"],
        ["--suffix", "a/b"],
        ["--namespace", "a,b"],
        ["--suffix", ""],
    ],
)
def test_conflicting_or_unsafe_namespaces_are_usage_errors(
    cli: str, flags: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        _parse(cli, *flags)
    assert exc.value.code == 2


def test_rag_input_takes_namespace_and_the_deprecated_suffix() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert RAGInput(query="q", namespace="tenant-a").namespace == "tenant-a"
    with pytest.warns(DeprecationWarning, match="suffix is deprecated"):
        old = RAGInput(query="q", suffix="tenant-a")
    assert old.namespace == "tenant-a" and "suffix" not in old.model_dump()
    with pytest.warns(DeprecationWarning, match="suffix is deprecated"):
        assert old.suffix == "tenant-a"
    with pytest.warns(DeprecationWarning):
        assert RAGInput(query="q", suffix="a", namespace="a").namespace == "a"
    with pytest.raises(ValidationError) as exc:
        RAGInput(query="q", suffix="a", namespace="b")
    assert "name different namespaces" in str(exc.value)
    with pytest.raises(ValidationError):
        RAGInput(query="q", namespace="../etc")


def test_a_deprecated_name_read_outside_pytest_warns_fails_the_suite() -> None:
    # The suite's filterwarnings turn the package's deprecations into
    # errors, so an internal read of a deprecated name (a filtered query
    # read RAGInput.suffix) fails whichever test reaches it.
    with pytest.raises(PackageDeprecationWarning, match="suffix is deprecated"):
        _ = RAGInput(query="q", namespace="tenant-a").suffix
    assert issubclass(PackageDeprecationWarning, DeprecationWarning)


def test_the_inspection_functions_take_namespace_and_the_deprecated_suffix() -> None:
    from unified_kg_rag import describe_namespace, export_graph
    from unified_kg_rag.domain.models.namespace import namespace_argument

    for function in (describe_namespace, export_graph):
        parameters = inspect.signature(function).parameters
        assert {"namespace", "suffix"} <= set(parameters), function
        assert parameters["suffix"].kind is inspect.Parameter.KEYWORD_ONLY
    assert namespace_argument("a", None) == "a"
    with pytest.warns(DeprecationWarning, match="suffix is deprecated"):
        assert namespace_argument(None, "a") == "a"
    with pytest.raises(ValueError, match="name different namespaces"):
        namespace_argument("a", "b")


def test_an_evaluation_item_s_suffix_is_its_namespace() -> None:
    with pytest.warns(DeprecationWarning):
        query = EvaluationQuery(query_id="1", question="q", metadata={"suffix": "v2"})
    assert query.metadata == {"namespace": "v2"}


def test_the_config_key_loads_from_a_file_and_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.safe_dump({"processing": {"document_parsing": {"index_value": "v2"}}})
    )
    assert (
        ConfigLoader(config_file).load_config().processing.document_parsing.namespace
        == "v2"
    )
    assert any(
        "processing.document_parsing.index_value is deprecated" in r.getMessage()
        for r in caplog.records
    )
    monkeypatch.setenv(env_var("processing.document_parsing.index_value"), "v3")
    assert ConfigLoader().load_config().processing.document_parsing.namespace == "v3"


def test_the_config_refuses_the_old_and_new_key_at_different_values() -> None:
    parsing = {"index_value": "a", "namespace": "b"}
    with pytest.raises(ValidationError, match="index_value"):
        Config(processing={"document_parsing": parsing})
    same = {"index_value": "a", "namespace": "a"}
    assert (
        Config(
            processing={"document_parsing": same}
        ).processing.document_parsing.namespace
        == "a"
    )


def test_the_old_and_new_variable_at_different_values_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # RV4-7: the new variable won with a WARNING, where every other surface
    # (a flag pair, a file pair, RAGInput, the console) refuses a conflict.
    from unified_kg_rag.shared.exceptions import ConfigurationError

    monkeypatch.setenv(env_var("processing.document_parsing.index_value"), "a")
    monkeypatch.setenv(env_var("processing.document_parsing.namespace"), "b")
    with pytest.raises(ConfigurationError, match="to different values"):
        ConfigLoader().load_config()
    monkeypatch.setenv(env_var("processing.document_parsing.namespace"), "a")
    assert ConfigLoader().load_config().processing.document_parsing.namespace == "a"


def test_the_config_attribute_index_value_reads_and_sets_the_namespace() -> None:
    parsing = Config().processing.document_parsing
    with pytest.warns(DeprecationWarning, match="index_value is deprecated"):
        parsing.index_value = "v2"
    assert parsing.namespace == "v2"
    with pytest.warns(DeprecationWarning):
        assert parsing.index_value == "v2"
    with pytest.warns(DeprecationWarning), pytest.raises(ValidationError):
        parsing.index_value = "Bad"
    assert parsing.namespace == "v2"


def _console_models() -> tuple[Callable[..., object], Callable[..., object]]:
    pytest.importorskip("fastapi")
    from unified_kg_rag.application.cli.console.ask import AskRequest
    from unified_kg_rag.application.cli.console.server import IngestBody

    return AskRequest, IngestBody


@pytest.mark.parametrize(("model", "old"), [(0, "suffix"), (1, "index_value")])
def test_the_console_requests_take_namespace_and_the_old_name(
    model: int, old: str
) -> None:
    cls = _console_models()[model]
    fields = {"query": "q"} if model == 0 else {"corpus": "workspace:docs"}
    assert cls(**fields, namespace="v2").namespace == "v2"  # type: ignore[attr-defined]
    with pytest.warns(DeprecationWarning, match=f"{old} is deprecated"):
        assert cls(**fields, **{old: "v2"}).namespace == "v2"  # type: ignore[attr-defined]
    with pytest.raises(ValidationError, match="name different namespaces"):
        cls(**fields, **{old: "v2", "namespace": "v3"})


def test_no_cli_defines_the_old_flag_outside_the_one_mapping() -> None:
    # A --suffix flag defined anywhere else would be a second mapping of the
    # namespace's names, free to drift from this one.
    defined = re.compile(r"[\"']--suffix[\"']")
    offenders = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "unified_kg_rag").rglob("*.py")
        if path.name != "namespace.py"
        and defined.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
    assert set(DEPRECATED_NAMESPACE_NAMES.values()) == {"--namespace", "namespace"}
