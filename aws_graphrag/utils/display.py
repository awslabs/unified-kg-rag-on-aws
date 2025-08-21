from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aws_graphrag.models import (
    Claim,
    Community,
    CommunityReport,
    Entity,
    PipelineContext,
    PipelineMetrics,
    PipelineStageResult,
    PipelineStageStatus,
    Relationship,
)

console = Console()

_SUMMARY_METRICS_CONFIG = {
    "Documents Processed": ("documents", "Source documents loaded into the system"),
    "Text Units Created": ("text_units", "Text segments created through chunking"),
    "Translated Units": (
        "translated_units",
        "Text segments processed through translation",
    ),
    "Total Entities": ("resolved_entities", "Unique entities identified and resolved"),
    "Total Relationships": (
        "resolved_relationships",
        "Relationships discovered between entities",
    ),
    "Total Claims": ("resolved_claims", "Claims extracted and entity-resolved"),
    "Communities Detected": (
        "communities",
        "Entity communities detected through clustering",
    ),
    "Community Reports": (
        "community_reports",
        "AI-generated community summary reports",
    ),
}

_STATUS_MAP = {
    PipelineStageStatus.COMPLETED: "[green]✓ COMPLETED[/green]",
    PipelineStageStatus.CACHED: "[blue]📋 CACHED[/blue]",
    PipelineStageStatus.FAILED: "[red]✗ FAILED[/red]",
    PipelineStageStatus.SKIPPED: "[yellow]⊘ SKIPPED[/yellow]",
}


def display_ascii_art(version: str = "1.0.0") -> None:
    ascii_art = f"""
[bold blue]
   ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗     ██████╗  █████╗  ██████╗
  ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║     ██╔══██╗██╔══██╗██╔════╝
  ██║  ███╗██████╔╝███████║██████╔╝███████║     ██████╔╝███████║██║  ███╗
  ██║   ██║██╔══██╗██╔══██║██╔══██╗██╔══██║     ██╔══██╗██╔══██║██║   ██║
  ╚██████╔╝██║  ██║██║  ██║██║  ██║██║  ██║     ██║  ██║██║  ██║╚██████╔╝
   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
[/bold blue]

[dim]Built by AWS Professional Services | Version {version}[/dim]
"""
    console.print(Panel(ascii_art, expand=False, border_style="dim"))


def display_pipeline_results(context: PipelineContext, limit: int = 10) -> None:
    display_pipeline_summary(context)
    display_stage_results(getattr(context, "stage_results", []))

    entities = _get_resolved_data(context, "entities")
    relationships = _get_resolved_data(context, "relationships")
    claims = _get_resolved_data(context, "claims")
    communities = getattr(context, "communities", None)
    community_reports = getattr(context, "community_reports", None)

    display_sample_entities(entities, limit=limit)
    display_sample_relationships(relationships, limit=limit)
    display_sample_claims(claims, limit=limit)
    display_communities(communities, community_reports, entities, claims, limit=limit)
    display_performance_summary(
        getattr(context, "stage_results", []), getattr(context, "global_metrics", None)
    )


def _get_resolved_data(context: PipelineContext, data_type: str) -> list | None:
    resolved_attr = f"resolved_{data_type}"
    return getattr(context, resolved_attr, getattr(context, data_type, None))


def display_pipeline_summary(context: PipelineContext) -> None:
    console.print("\n" + "=" * 80)
    console.print(
        Panel.fit(
            "[bold blue]Pipeline Results Summary[/bold blue]", border_style="blue"
        )
    )

    summary_table = Table(
        title="Processing Summary", show_header=True, header_style="bold magenta"
    )
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Count", justify="right", style="green")
    summary_table.add_column("Details", style="yellow")

    for metric, (context_attr, details) in _SUMMARY_METRICS_CONFIG.items():
        data = getattr(context, context_attr, None)
        count = len(data) if data else 0
        if count > 0 or metric in ["Documents Processed", "Text Units Created"]:
            summary_table.add_row(metric, str(count), details)

    console.print(summary_table)


def display_stage_results(stage_results: list[PipelineStageResult]) -> None:
    if not stage_results:
        return

    console.print("\n" + "=" * 80)
    console.print(
        Panel.fit(
            "[bold green]Pipeline Stage Execution Details[/bold green]",
            border_style="green",
        )
    )

    stage_table = Table(
        title="Stage Execution Summary",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    stage_table.add_column("Stage", style="cyan", no_wrap=True, min_width=20)
    stage_table.add_column("Status", justify="center", min_width=15)
    stage_table.add_column("Duration", justify="right", style="blue", min_width=12)
    stage_table.add_column("Input", justify="right", style="yellow", min_width=10)
    stage_table.add_column("Output", justify="right", style="green", min_width=10)

    for result in stage_results:
        status = _STATUS_MAP.get(result.status, f"[dim]{result.status.value}[/dim]")
        duration = (
            f"{result.duration_seconds:.2f}s"
            if result.duration_seconds is not None
            else "N/A"
        )
        stage_table.add_row(
            result.stage_name.replace("_", " ").title(),
            status,
            duration,
            str(result.input_count),
            str(result.output_count),
        )
    console.print(stage_table)


def display_sample_entities(entities: list[Entity] | None, limit: int = 10) -> None:
    if not entities:
        return

    console.print("\n" + "=" * 80)
    console.print(
        Panel.fit(
            f"[bold magenta]Sample Extracted Entities (Top {min(limit, len(entities))})[/bold magenta]",
            border_style="magenta",
        )
    )

    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Entity Name", style="cyan", min_width=25)
    table.add_column("Type", style="green", min_width=15)
    table.add_column("Description", style="yellow", max_width=60, min_width=30)

    for entity in entities[:limit]:
        table.add_row(
            str(getattr(entity, "name", "Unknown")),
            str(getattr(entity, "type", "Unknown")),
            _truncate_text(getattr(entity, "description", ""), 150),
        )
    console.print(table)


def _truncate_text(text: str | None, max_length: int) -> str:
    if not text:
        return ""
    return text[: max_length - 3] + "..." if len(text) > max_length else text


def display_sample_relationships(
    relationships: list[Relationship] | None, limit: int = 10
) -> None:
    if not relationships:
        return

    console.print("\n" + "=" * 80)
    console.print(
        Panel.fit(
            f"[bold cyan]Sample Relationships (Top {min(limit, len(relationships))})[/bold cyan]",
            border_style="cyan",
        )
    )

    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Source", style="cyan", min_width=25)
    table.add_column("Relationship", style="green", justify="center", min_width=20)
    table.add_column("Target", style="cyan", min_width=25)
    table.add_column("Description", style="dim", max_width=70, min_width=40)

    for rel in relationships[:limit]:
        source_name = getattr(rel, "source_name", "Unknown")
        target_name = getattr(rel, "target_name", "Unknown")

        table.add_row(
            _truncate_text(source_name, 30),
            str(getattr(rel, "type", "Unknown")),
            _truncate_text(target_name, 30),
            _truncate_text(getattr(rel, "description", ""), 120),
        )
    console.print(table)


def display_sample_claims(claims: list[Claim] | None, limit: int = 10) -> None:
    if not claims:
        return

    console.print("\n" + "=" * 80)
    console.print(
        Panel.fit(
            f"[bold yellow]Sample Claims (Top {min(limit, len(claims))})[/bold yellow]",
            border_style="yellow",
        )
    )

    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Subject", style="cyan", max_width=30, min_width=20)
    table.add_column("Object", style="green", max_width=30, min_width=20)
    table.add_column("Status", style="blue", justify="center", min_width=12)
    table.add_column("Claim", style="yellow", max_width=80, min_width=50)

    for claim in claims[:limit]:
        subject_name = getattr(claim, "subject_name", "Unknown")
        obj_name = getattr(claim, "object_name", "Unknown")
        claim_type = getattr(claim, "type", "Unknown")
        description = getattr(claim, "description", "")
        table.add_row(
            _truncate_text(subject_name, 30),
            _truncate_text(obj_name, 30),
            str(claim_type),
            _truncate_text(description, 150),
        )
    console.print(table)


def display_communities(
    communities: list[Community] | None,
    community_reports: list[CommunityReport] | None,
    entities: list[Entity] | None,
    claims: list[Claim] | None,
    limit: int = 5,
) -> None:
    if not communities:
        return

    entity_lookup = _create_entity_and_claim_lookup(entities or [], claims or [])
    reports_lookup = _create_reports_lookup(community_reports or [])

    console.print("\n" + "=" * 80)
    console.print(
        Panel.fit(
            f"[bold red]Detected Communities (Top {min(limit, len(communities))})[/bold red]",
            border_style="red",
        )
    )

    for i, community in enumerate(communities[:limit], 1):
        community_id = getattr(community, "id", f"community_{i}")
        title = getattr(community, "name", f"Community {i}")
        entity_ids = getattr(community, "entity_ids", [])
        size = getattr(community, "size", len(entity_ids))

        console.print(f"\n[bold]Community {i}:[/bold] {title}")
        console.print(f"[dim]ID:[/dim] {community_id}")
        console.print(f"[dim]Size:[/dim] {size} entities")

        if entity_ids:
            entity_names = [
                entity_lookup.get(eid, f"{eid[:12]}...") for eid in entity_ids[:5]
            ]
            if len(entity_ids) > 5:
                entity_names.append(f"... and {len(entity_ids) - 5} more")
            console.print(f"[dim]Entities:[/dim] {', '.join(entity_names)}")

        if community_id in reports_lookup:
            report = reports_lookup[community_id]
            summary = getattr(report, "summary", "")
            if summary:
                console.print(
                    f"[dim]Summary (Rank {getattr(report, 'rank', 1)}):[/dim] {_truncate_text(summary, 200)}"
                )


def _create_entity_and_claim_lookup(
    entities: list[Entity], claims: list[Claim] | None = None
) -> dict[str, str]:
    lookup = {}

    for entity in entities:
        entity_id = getattr(entity, "id", None)
        entity_name = getattr(entity, "name", None)

        if entity_id and entity_name:
            lookup[entity_id] = entity_name
        elif entity_id:
            lookup[entity_id] = entity_id[:8]

    if claims:
        for claim in claims:
            claim_id = getattr(claim, "id", None)
            claim_object_name = getattr(claim, "object_name", None)
            claim_subject_name = getattr(claim, "subject_name", None)

            if claim_id and claim_object_name and claim_subject_name:
                lookup[claim_id] = f"{claim_subject_name} -> {claim_object_name}"
            elif claim_id:
                lookup[claim_id] = claim_id[:8]

    return lookup


def _create_reports_lookup(
    community_reports: list[CommunityReport],
) -> dict[str, CommunityReport]:
    return {r.community_id: r for r in community_reports if hasattr(r, "community_id")}


def display_performance_summary(
    stage_results: list[PipelineStageResult], metrics: PipelineMetrics | None
) -> None:
    if not stage_results:
        return

    console.print("\n" + "=" * 80)
    console.print(
        Panel.fit(
            "[bold yellow]Performance Summary[/bold yellow]", border_style="yellow"
        )
    )

    metrics_table = _create_performance_metrics_table(stage_results, metrics)

    if metrics_table.row_count > 0:
        console.print(metrics_table)


def _create_performance_metrics_table(
    stage_results: list[PipelineStageResult], metrics: PipelineMetrics | None
) -> Table:
    metrics_table = Table(show_header=False, show_edge=False, box=None)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green", justify="right")

    total_duration = sum(r.duration_seconds or 0 for r in stage_results)
    completed_stages = sum(
        1 for r in stage_results if r.status == PipelineStageStatus.COMPLETED
    )
    cached_stages = sum(
        1 for r in stage_results if r.status == PipelineStageStatus.CACHED
    )

    metrics_table.add_row("Total Pipeline Duration", f"{total_duration:.2f} seconds")
    metrics_table.add_row("Stages Executed", str(completed_stages))
    metrics_table.add_row("Stages from Cache", str(cached_stages))

    if metrics:
        _add_pipeline_metrics(metrics_table, metrics)

    return metrics_table


def _add_pipeline_metrics(metrics_table: Table, metrics: PipelineMetrics) -> None:
    metric_mappings = [
        ("cache_hit_rate", "Cache Hit Rate", lambda x: f"{x:.2%}"),
        (
            "gleaning_improvement_rate",
            "Gleaning Improvement Rate",
            lambda x: f"{x:.2%}",
        ),
        ("entity_resolution_merge_rate", "Entity Merge Rate", lambda x: f"{x:.2%}"),
        (
            "relationship_resolution_merge_rate",
            "Relationship Merge Rate",
            lambda x: f"{x:.2%}",
        ),
        ("claim_resolution_merge_rate", "Claim Merge Rate", lambda x: f"{x:.2%}"),
        ("community_modularity_score", "Community Modularity", lambda x: f"{x:.2f}"),
    ]

    for attr_name, display_name, formatter in metric_mappings:
        if hasattr(metrics, attr_name):
            value = getattr(metrics, attr_name)
            metrics_table.add_row(display_name, formatter(value))

    throughputs = getattr(metrics, "stage_throughput", {})
    if throughputs:
        avg_throughput = sum(throughputs.values()) / len(throughputs)
        metrics_table.add_row("Avg. Throughput", f"{avg_throughput:.2f} items/sec")
