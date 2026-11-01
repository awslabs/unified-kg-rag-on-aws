# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI for automatic prompt tuning (MS GraphRAG prompt_tune, AWS-native).

Samples documents from a directory, profiles the corpus domain/language/persona/
entity-types via Bedrock, and writes domain-adapted ``custom_prompts`` as a YAML
fragment the user reviews and merges into their config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from unified_kg_rag.adapters.ingestion.parser import ParserFactory
from unified_kg_rag.application.cli.config_args import (
    add_config_arguments,
    exit_on_config_error,
    load_cli_config,
)
from unified_kg_rag.application.cli.conventions import (
    EXIT_FAILURE,
    EXIT_INTERRUPTED,
    EXIT_USAGE,
    ArgumentParser,
    add_output_format_argument,
    exits_on_sigterm,
    failed,
    json_output,
    print_json,
    stoppable,
    usage_error,
)
from unified_kg_rag.application.cli.help_text import doc_epilog
from unified_kg_rag.application.cli.preflight import missing_extras_error
from unified_kg_rag.application.cli.provider_args import (
    add_provider_arguments,
    apply_provider_arguments,
)
from unified_kg_rag.application.cli.query_args import positive_int
from unified_kg_rag.application.config import load_cli_dotenv
from unified_kg_rag.application.prompts.tuner import PromptTuner
from unified_kg_rag.domain.ingestion.cache_keys import is_excluded_source_path
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import ConfigurationError, get_logger, setup_logging
from unified_kg_rag.shared.utils import event_loop

logger = get_logger(__name__)

# Read as-is: plain text needs no loader (and .md would otherwise require the
# optional `unstructured` package). Every other format goes through the same
# ParserFactory loaders as ingestion (PDF, CSV, JSON, registered custom ones).
_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


def load_corpus_texts(
    source_dir: Path, max_docs: int, config: Config | None = None
) -> list[str]:
    """Read up to ``max_docs`` documents from ``source_dir`` as text."""
    config = config or Config()
    supported = _TEXT_SUFFIXES | set(ParserFactory.get_supported_extensions())
    texts: list[str] = []
    for path in sorted(source_dir.rglob("*")):
        suffix = path.suffix.lower()
        if (
            not path.is_file()
            or suffix not in supported
            or is_excluded_source_path(path, source_dir)
        ):
            continue
        try:
            if suffix in _TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8")
            else:
                document = ParserFactory.create_parser(path, config).parse_file(path)
                text = document.page_content
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
            continue
        if text.strip():
            texts.append(text)
        if len(texts) >= max_docs:
            break
    return texts


def _build_parser() -> argparse.ArgumentParser:
    parser = ArgumentParser(
        allow_abbrev=False,
        description="Generate domain-adapted custom_prompts from a corpus sample.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=doc_epilog(
            "8-prompt-tuning-run-prompt-tuning",
            "run-prompt-tuning --source-directory ./source --output-path "
            "tuned_prompts.yaml",
        ),
    )
    parser.add_argument(
        "--source-directory",
        required=True,
        type=Path,
        help=(
            "Directory of documents to sample (.txt/.md plus every format "
            "run-ingestion parses, e.g. .pdf)"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("tuned_prompts.yaml"),
        help="Where to write the tuned custom_prompts YAML",
    )
    parser.add_argument(
        "--max-docs",
        type=positive_int,
        default=20,
        help="Max documents to sample (at least 1)",
    )
    add_config_arguments(parser)
    add_output_format_argument(parser)
    add_provider_arguments(parser)
    return parser


@exits_on_sigterm
def main() -> int:
    load_cli_dotenv()
    args = _build_parser().parse_args()
    as_json = json_output(args)
    try:
        with exit_on_config_error():
            config = apply_provider_arguments(load_cli_config(args), args)
        if not args.source_directory.is_dir():
            usage_error(f"source directory not found: '{args.source_directory}'")
        if error := missing_extras_error(config):
            usage_error(error)
        setup_logging(config, stream=sys.stderr if as_json else None)

        texts = load_corpus_texts(args.source_directory, args.max_docs, config)
        if not texts:
            error = f"No text documents found under '{args.source_directory}'"
            logger.error("%s", error)
            return failed(EXIT_FAILURE, error)

        tuner = PromptTuner(config)
        result = event_loop.run(
            stoppable(tuner.tune(texts)), io_workers=config.processing.io_workers
        )

        args.output_path.write_text(
            yaml.safe_dump(result, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        logger.info("Wrote tuned prompts to '%s'", args.output_path)
        logger.info("Detected domain: '%s'", result["profile"]["domain"])
        if as_json:
            print_json(
                {"output_path": str(args.output_path), "profile": result["profile"]}
            )
        return 0
    except KeyboardInterrupt:
        logger.warning("Prompt tuning interrupted by user.")
        # Raised, not returned: exits_on_sigterm prints the JSON status.
        raise SystemExit(EXIT_INTERRUPTED) from None
    except ConfigurationError as e:
        logger.error("Prompt tuning configuration error: %s", e)
        return failed(EXIT_USAGE, e)
    except Exception as e:
        # Surface a clean error line instead of a raw traceback (parity with the
        # other CLIs); details still go to the log at debug level.
        logger.error("Prompt tuning failed: %s", e)
        logger.debug("Traceback:", exc_info=True)
        return failed(EXIT_FAILURE, e)


if __name__ == "__main__":
    sys.exit(main())
