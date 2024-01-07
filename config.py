from dataclasses import dataclass


@dataclass
class Files:
    train_data: str
    train_targets: str
    test_data: str
    test_targets: str
    target_names: str
    vectorizer: str
    model: str


@dataclass
class Params:
    iterations: int = 100
    learning_rate: float | None = None
    tree_depth: int | None = None


@dataclass
class Urls:
    mlflow_url: str = "http://128.0.1.1:8080"


@dataclass
class MainConfig:
    files: Files
    params: Params
    urls: Urls
