from typing import Any


class MetricsMixin:
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._metrics: dict[str, Any] = {"timings": {}, "metrics": {}}

    def get_metrics(self) -> dict[str, Any]:
        return self._metrics.copy()

    def reset_metrics(self) -> None:
        self._metrics = {"timings": {}, "metrics": {}}

    def _record_metric(self, key: str, value: Any) -> None:
        self._metrics["metrics"][key] = value

    def _record_timing(self, operation: str, duration: float) -> None:
        self._metrics["timings"][operation] = duration
