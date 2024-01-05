from dataclasses import dataclass


@dataclass
class TrainMainConfig:
    iterations: int = 100
    learning_rate: float | None = None
    tree_depth: int | None = None
