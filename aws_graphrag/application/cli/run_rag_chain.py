# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import json
import sys
import uuid
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import nest_asyncio
from dotenv import load_dotenv
from rich.panel import Panel

from aws_graphrag.domain.models import Constants, SearchStrategy, SearchType
from aws_graphrag.retrieval import ChainMode, GraphRAGChain, RAGInput, create_rag_chain
from aws_graphrag.shared import get_config, get_logger
from aws_graphrag.shared.utils import console, display_ascii_art

ROOT_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIRECTORY))

load_dotenv()
nest_asyncio.apply()
logger = get_logger(__name__)

try:
    __version__ = version("aws-graphrag")
except (FileNotFoundError, ImportError, ValueError):
    __version__ = "unknown"


class CommandLineInterface:
    def __init__(self) -> None:
        self.parser = self._setup_arguments()

    @staticmethod
    def _setup_arguments() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="GraphRAG Chain - Retrieval-augmented generation using knowledge graphs",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        parser.add_argument(
            "--query", "-q", help="Execute a single query against the knowledge graph"
        )
        parser.add_argument(
            "--interactive",
            "-i",
            action="store_true",
            help="Launch interactive chat mode for continuous conversation",
        )
        parser.add_argument(
            "--mode",
            default=ChainMode.RAG.value,
            choices=[cm.value for cm in ChainMode],
            help="Set the execution mode: 'rag' for full generation, 'search' for retrieval only",
        )
        parser.add_argument(
            "--conversation-id",
            help="Continue an existing conversation using the specified conversation ID",
        )
        parser.add_argument(
            "--use-memory",
            action="store_true",
            help="Enable conversation memory (auto-enabled in interactive mode)",
        )
        parser.add_argument(
            "--suffix",
            help="Suffix for multi-tenant or versioned indices",
        )
        parser.add_argument(
            "--enable-thinking",
            action="store_true",
            help="Enable thinking mode for language model reasoning and step-by-step problem solving",
        )
        parser.add_argument(
            "--search-strategy",
            default="auto",
            choices=[ss.value for ss in SearchStrategy],
            help="Choose the search strategy",
        )
        parser.add_argument(
            "--search-type",
            default="hybrid",
            choices=[st.value for st in SearchType],
            help="Specify the search method",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=10,
            help="Set the maximum number of search results",
        )
        parser.add_argument(
            "--retrieval-multiplier",
            type=int,
            default=1,
            help="Set the retrieval multiplier for increasing search depth",
        )
        parser.add_argument(
            "--disable-query-processing",
            action="store_true",
            help="Skip automatic query processing",
        )
        parser.add_argument(
            "--filters",
            nargs="*",
            help="Apply attribute filters using 'key:value' format",
        )
        parser.add_argument(
            "--output-format",
            default="text",
            choices=["text", "json"],
            help="Select output format",
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Display detailed information",
        )
        parser.add_argument(
            "--config-path",
            type=str,
            help="Path to custom configuration file",
        )
        return parser

    def parse_args(self) -> argparse.Namespace:
        return self.parser.parse_args()


class RAGChainRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = get_config(Path(args.config_path) if args.config_path else None)
        self.rag_chain: GraphRAGChain | None = None
        self.interactive_commands: dict[str, Callable] = {
            "help": self._handle_help,
            "new": self._handle_new_conversation,
            "set-filter": self._handle_set_filter,
            "clear-filters": self._handle_clear_filters,
            "show-config": self._handle_show_config,
        }
        self._validate_args()

    def _validate_args(self) -> None:
        if not self.args.interactive and not self.args.query:
            console.print(
                "[red]Error: Either --query or --interactive option is required.[/red]"
            )
            sys.exit(1)

    def _get_suffix_display(self) -> str:
        if self.args.suffix:
            return (
                f"[dim]Index/Label Suffix:[/dim] [orange3]{self.args.suffix}[/orange3]"
            )

        try:
            default_suffix = Constants.DEFAULT_SUFFIX.value
            if self.config.indexing.additional_suffix:
                final_suffix = (
                    f"{default_suffix}-{self.config.indexing.additional_suffix}"
                )
                return f"[dim]Index/Label Suffix:[/dim] [gray]{final_suffix}[/gray]"
            return f"[dim]Index/Label Suffix:[/dim] [gray]{default_suffix}[/gray]"
        except Exception:
            return f"[dim]Index/Label Suffix:[/dim] [gray]{Constants.DEFAULT_SUFFIX.value}[/gray]"

    async def run(self) -> None:
        display_ascii_art(version=__version__)
        self._display_run_info()
        await self._initialize_chain()

        if self.args.interactive:
            await self._interactive_mode()
        else:
            rag_input = RAGInput(
                query=self.args.query,
                suffix=self.args.suffix,
                enable_thinking=self.args.enable_thinking,
                search_strategy=self.args.search_strategy,
                search_type=SearchType(self.args.search_type),
                conversation_id=self.args.conversation_id,
                use_memory=self.args.use_memory,
                enable_query_processing=not self.args.disable_query_processing,
                target_language=self.config.processing.translation.target_language,
                top_k=self.args.top_k,
                retrieval_multiplier=self.args.retrieval_multiplier,
                filters=self._parse_filters(self.args.filters),
            )

            result = await self._run_query(rag_input)
            if self.args.output_format == "json":
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                self._print_result(result, self.args.verbose)

            if not result.get("success", False):
                sys.exit(1)

    async def _initialize_chain(self) -> None:
        console.print("[bold]Initializing RAG chain...[/bold]")
        try:
            self.rag_chain = await create_rag_chain(
                config=self.config,
                mode=ChainMode(self.args.mode),
                enable_thinking=self.args.enable_thinking,
            )
            console.print("[green]GraphRAG chain initialized successfully![/green]")
        except Exception as e:
            logger.exception("Failed to initialize RAG chain: %s", e)
            console.print(f"[red]Error during initialization: {e}[/red]")
            sys.exit(1)

    async def _interactive_mode(self) -> None:
        console.print(
            "[bold blue]\nGraph RAG Interactive Mode[/bold blue]\nType 'quit', 'exit' to end, or 'help' for commands."
        )
        console.print("---")

        state = {
            "conversation_id": self.args.conversation_id or str(uuid.uuid4()),
            "filters": self._parse_filters(self.args.filters) or {},
        }

        while True:
            try:
                raw_input = console.input("\n[bold cyan]Question:[/bold cyan] ").strip()
                if not raw_input:
                    continue
                if raw_input.lower() in ["quit", "exit"]:
                    console.print("[yellow]Ending conversation.[/yellow]")
                    break

                parts = raw_input.split()
                command, cmd_args = parts[0].lower(), parts[1:]

                if handler := self.interactive_commands.get(command):
                    if command == "new":
                        state["conversation_id"] = handler()
                    else:
                        handler(state=state, args=cmd_args)
                    continue
                rag_input = RAGInput(
                    query=raw_input,
                    suffix=self.args.suffix,
                    enable_thinking=self.args.enable_thinking,
                    search_strategy=self.args.search_strategy,
                    search_type=self.args.search_type,
                    conversation_id=(
                        str(state["conversation_id"])
                        if state["conversation_id"] is not None
                        else None
                    ),
                    use_memory=True,
                    filters=(
                        state["filters"] if isinstance(state["filters"], dict) else None
                    ),
                    top_k=self.args.top_k,
                    retrieval_multiplier=self.args.retrieval_multiplier,
                    enable_query_processing=not self.args.disable_query_processing,
                )
                if self.rag_chain:
                    self.rag_chain.mode = ChainMode.RAG

                result = await self._run_query(rag_input)
                self._print_result(result, verbose=True)
                state["conversation_id"] = result.get(
                    "conversation_id", state["conversation_id"]
                )

            except KeyboardInterrupt:
                console.print("\n[yellow]Conversation interrupted.[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Unexpected error: {e}[/red]")
                logger.error(
                    "Unexpected error in interactive mode: %s", e, exc_info=True
                )

    async def _run_query(self, rag_input: RAGInput) -> dict[str, Any]:
        if not self.rag_chain:
            raise RuntimeError("RAG chain not initialized")
        try:
            console.print(
                f"[bold]Executing RAG query in '{self.args.mode}' mode...[/bold]",
                "[dim]This may take some time depending on query complexity[/dim]",
            )
            result = await self.rag_chain.ainvoke(rag_input)
            console.print("\n[bold green]Query executed successfully![/bold green]")

            if isinstance(result, dict):
                return {"success": True, **result}
            else:
                return {"success": True, **result.model_dump()}

        except Exception as e:
            logger.exception("Error during query execution: %s", e)
            console.print(f"\n[bold red]Error executing query: {e}[/bold red]")
            return {
                "success": False,
                "error": str(e),
                "conversation_id": rag_input.conversation_id,
            }

    def _display_run_info(self) -> None:
        config_lines = [
            "[bold blue]RAG Configuration[/bold blue]",
            "[bold]Current Configuration[/bold]",
            f"[dim]Execution Mode:[/dim] {self.args.mode}",  # 모드 정보 표시
            f"[dim]Search Strategy:[/dim] {self.args.search_strategy}",
            f"[dim]Search Type:[/dim] {self.args.search_type}",
            f"[dim]Top K Results:[/dim] {self.args.top_k}",
            f"[dim]Retrieval Multiplier:[/dim] {self.args.retrieval_multiplier}",
            f"[dim]Use Memory:[/dim] {self.args.use_memory or self.args.interactive}",
            f"[dim]Query Processing:[/dim] {not self.args.disable_query_processing}",
            f"[dim]Verbose Output:[/dim] {self.args.verbose}",
            self._get_suffix_display(),
        ]

        if self.args.filters:
            config_lines.append(f"[dim]Filters:[/dim] {', '.join(self.args.filters)}")

        if self.args.conversation_id:
            config_lines.append(
                f"[dim]Conversation ID:[/dim] {self.args.conversation_id}"
            )

        if self.args.query:
            config_lines.append(f"[dim]Query:[/dim] {self.args.query}")

        console.print(Panel.fit("\n".join(config_lines), border_style="blue"))

    @staticmethod
    def _parse_filters(filter_args: list[str] | None) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if not filter_args:
            return filters
        for item in filter_args:
            if ":" in item:
                key, value = item.split(":", 1)
                filters[key] = value
            else:
                logger.warning(
                    "Invalid filter format skipped: '%s'. Expected 'key:value'.", item
                )
        return filters

    @staticmethod
    def _print_result(result: dict[str, Any], verbose: bool = False) -> None:
        if not result.get("success", False):
            console.print(f"\n[red]Error: {result.get('error', 'Unknown error')}[/red]")
            return

        if answer := result.get("answer"):
            console.print("\n\n" + "=" * 50)
            console.print("[bold blue]Final Answer:[/bold blue]")
            console.print(answer)
            console.print("=" * 50)
        else:
            console.print("\n\n" + "=" * 50)
            console.print("[bold blue]Search Mode Results[/bold blue]")
            console.print("=" * 50)

        if verbose:
            console.print("\n[bold]Details:[/bold]")

            if pq := result.get("processed_query"):
                pq_panel_content = ""

                if original_query := pq.get("original_query"):
                    pq_panel_content += f"  - Original Query: {original_query}\n"

                if translated_query := pq.get("translated_query"):
                    pq_panel_content += f"  - Translated Query: {translated_query}\n"

                if entities := pq.get("entities"):
                    pq_panel_content += (
                        f"  - Extracted Entities: {len(entities)} found\n"
                    )
                    if entities and len(entities) <= 5:
                        pq_panel_content += f"    {', '.join(entities)}\n"

                if pq_panel_content:
                    console.print(
                        Panel(
                            pq_panel_content.rstrip(),
                            title="[cyan]Query Processing Info[/cyan]",
                            border_style="cyan",
                        )
                    )

            sources = result.get("sources")
            if not sources and result.get("search_results"):
                sources = result.get("search_results", {}).get("results", [])

            if sources:
                sources_content = ""
                for i, source in enumerate(sources[:5], 1):
                    score = source.get("score", 0)
                    source_id = source.get("source", "N/A")
                    sources_content += f"  {i}. {source_id} (score: {score:.3f})\n"

                if len(sources) > 5:
                    sources_content += f"  ... and {len(sources) - 5} more sources\n"

                console.print(
                    Panel(
                        sources_content.rstrip(),
                        title=f"[cyan]Reference Sources ({len(sources)} found)[/cyan]",
                        border_style="cyan",
                    )
                )

            search_result = result.get("search_results", {})
            metadata = result.get("metadata", {})

            metrics_content = ""
            all_metadata = {**search_result, **metadata}

            for key, value in all_metadata.items():
                if key in ["results", "query"]:
                    continue
                if key == "search_strategy":
                    metrics_content += f"  - Search Strategy: {value}\n"
                elif key == "processing_time":
                    metrics_content += f"  - Processing Time: {value:.2f}s\n"
                elif isinstance(value, int | float | str | bool):
                    metrics_content += f"  - {key.replace('_', ' ').title()}: {value}\n"
                elif isinstance(value, list | dict) and len(str(value)) < 100:
                    metrics_content += f"  - {key.replace('_', ' ').title()}: {value}\n"

            if metrics_content:
                console.print(
                    Panel(
                        metrics_content.rstrip(),
                        title="[cyan]Execution Metrics[/cyan]",
                        border_style="cyan",
                    )
                )

    @staticmethod
    def _handle_help(**kwargs: Any) -> None:
        console.print("[bold]Available commands:[/bold]")
        console.print("  quit/exit      - End the conversation")
        console.print("  new            - Start a new conversation")
        console.print("  help           - Show this help message")
        console.print(
            "  set-filter <k:v> - Add/update filter (e.g., set-filter entity_type:person)"
        )
        console.print("  clear-filters  - Remove all filters")
        console.print("  show-config    - Display current configuration")

    @staticmethod
    def _handle_new_conversation(**kwargs: Any) -> str:
        new_id = str(uuid.uuid4())
        console.print(
            f"[green]Starting new conversation. (ID: '{new_id[:8]}...')[/green]"
        )
        return new_id

    @staticmethod
    def _handle_set_filter(state: dict[str, Any], args: list[str]) -> None:
        if not args or ":" not in args[0]:
            console.print("[red]Invalid filter format. Use 'key:value'[/red]")
            return
        key, value = args[0].split(":", 1)
        state["filters"][key] = value
        console.print(f"[green]Filter added: '{key}' = '{value}'[/green]")

    @staticmethod
    def _handle_clear_filters(state: dict[str, Any], **kwargs: Any) -> None:
        state["filters"].clear()
        console.print("[green]All filters cleared.[/green]")

    def _handle_show_config(self, state: dict[str, Any], **kwargs: Any) -> None:
        config_lines = [
            "[bold]Current Configuration:[/bold]",
            "[dim]Execution Mode:[/dim] rag (Interactive mode)",
            f"[dim]Search Strategy:[/dim] {self.args.search_strategy}",
            f"[dim]Search Type:[/dim] {self.args.search_type}",
            f"[dim]Top K Results:[/dim] {self.args.top_k}",
            f"[dim]Retrieval Multiplier:[/dim] {self.args.retrieval_multiplier}",
            f"[dim]Use Memory:[/dim] {self.args.use_memory or self.args.interactive}",
            f"[dim]Query Processing:[/dim] {not self.args.disable_query_processing}",
            f"[dim]Verbose Output:[/dim] {self.args.verbose}",
            self._get_suffix_display(),
        ]

        if state["filters"]:
            filters_str = ", ".join([f"{k}:{v}" for k, v in state["filters"].items()])
            config_lines.append(f"[dim]Filters:[/dim] {filters_str}")
        else:
            config_lines.append("[dim]Filters:[/dim] None")

        console.print(Panel.fit("\n".join(config_lines), border_style="blue"))


async def async_main() -> None:
    try:
        cli = CommandLineInterface()
        args = cli.parse_args()
        runner = RAGChainRunner(args)
        await runner.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Execution interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]An unexpected error occurred: {e}[/red]")
        logger.exception("Unexpected error during execution")
        sys.exit(1)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
