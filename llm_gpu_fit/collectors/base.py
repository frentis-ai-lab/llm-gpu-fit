from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class CollectorResult:
    models: list[dict] = field(default_factory=list)
    benchmarks: list[dict] = field(default_factory=list)
    inference_perf: list[dict] = field(default_factory=list)


class BaseCollector(ABC):
    name: str = "base"

    @abstractmethod
    def collect(self) -> CollectorResult:
        ...
