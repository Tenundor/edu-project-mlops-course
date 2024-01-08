import json
from pathlib import Path

from dvc.api import DVCFileSystem


def _load_json_file(filename: str) -> list:
    fs = DVCFileSystem(".")
    path = Path("data") / filename
    with fs.open(str(path), "r") as file:
        data = json.load(file)
        return data


def load_news_data(filename: str) -> list[str]:
    return _load_json_file(filename)


def load_news_targets(filename: str) -> list[int]:
    return _load_json_file(filename)


def load_news_target_names(filename: str) -> list[str]:
    return _load_json_file(filename)
