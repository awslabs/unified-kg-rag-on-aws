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

from aws_graphrag.application.prompts.tuner import PromptTuner
from aws_graphrag.shared import get_config, get_logger

logger = get_logger(__name__)

_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


def load_corpus_texts(source_dir: Path, max_docs: int) -> list[str]:
    """Read plain-text documents from ``source_dir`` (up to ``max_docs``)."""
    texts: list[str] = []
    for path in sorted(source_dir.rglob("*")):
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        try:
            texts.append(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
        if len(texts) >= max_docs:
            break
    return texts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate domain-adapted custom_prompts from a corpus sample."
    )
    parser.add_argument(
        "--source-dir", required=True, type=Path, help="Directory of text documents"
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
    config = get_config(args.config_path)

    texts = load_corpus_texts(args.source_dir, args.max_docs)
    if not texts:
        logger.error("No text documents found under '%s'", args.source_dir)
        return 1

    tuner = PromptTuner(config)
    result = asyncio.run(tuner.tune(texts))

    args.output.write_text(
        yaml.safe_dump(result, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    logger.info("Wrote tuned prompts to '%s'", args.output)
    logger.info("Detected domain: '%s'", result["profile"]["domain"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
