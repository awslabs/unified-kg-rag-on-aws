# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI for automatic prompt tuning (MS GraphRAG prompt_tune, AWS-native).

Samples documents from a directory, profiles the corpus domain/language/persona/
entity-types via Bedrock, and writes domain-adapted ``custom_prompts`` as a YAML
fragment the user reviews and merges into their config.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

from unified_kg_rag.adapters.ingestion.parser import ParserFactory
from unified_kg_rag.application.prompts.tuner import PromptTuner
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import get_config, get_logger

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
        if not path.is_file() or suffix not in supported:
            continue
        try:
            if suffix in _TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8")
            else:
                document = ParserFactory.create_parser(path, config).parse_file(path)
                text = (document.content.text if document.content else None) or ""
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
            continue
        if text.strip():
            texts.append(text)
        if len(texts) >= max_docs:
            break
    return texts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate domain-adapted custom_prompts from a corpus sample."
    )
    parser.add_argument(
        # Canonical flag matches run-ingestion's --source-directory; --source-dir
        # is kept as a backward-compatible alias so existing invocations still work.
        "--source-directory",
        "--source-dir",
        dest="source_directory",
        required=True,
        type=Path,
        help=(
            "Directory of documents to sample (.txt/.md plus every format "
            "run-ingestion parses, e.g. .pdf)"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("tuned_prompts.yaml"), help="Output YAML"
    )
    parser.add_argument(
        "--max-docs", type=int, default=20, help="Max documents to sample"
    )
    parser.add_argument(
        "--config-path", type=str, default=None, help="Config YAML path"
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        config = get_config(args.config_path)

        texts = load_corpus_texts(args.source_directory, args.max_docs, config)
        if not texts:
            logger.error("No text documents found under '%s'", args.source_directory)
            return 1

        tuner = PromptTuner(config)
        result = asyncio.run(tuner.tune(texts))

        args.output.write_text(
            yaml.safe_dump(result, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        logger.info("Wrote tuned prompts to '%s'", args.output)
        logger.info("Detected domain: '%s'", result["profile"]["domain"])
        return 0
    except KeyboardInterrupt:
        logger.warning("Prompt tuning interrupted by user.")
        return 130
    except Exception as e:
        # Surface a clean error line instead of a raw traceback (parity with the
        # other CLIs); details still go to the log at debug level.
        logger.error("Prompt tuning failed: %s", e)
        logger.debug("Traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
