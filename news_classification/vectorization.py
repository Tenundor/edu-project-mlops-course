import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class NewsVectorizer:
    def __init__(self):
        self._vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            max_df=0.5,
            min_df=5,
            stop_words="english",
        )
        self._artifacts_dir_path = Path("models")

    def fit(self, data: list[str]):
        self._vectorizer.fit(data)

    def fit_transform(self, data: list[str]):
        return self._vectorizer.fit_transform(data)

    def transform(self, data: list[str]):
        return self._vectorizer.transform(data)

    def save_vectorizer(self, filename: str) -> None:
        self._artifacts_dir_path.mkdir(parents=True, exist_ok=True)
        path = self._artifacts_dir_path / filename
        data_to_save = {
            "vocabulary_": self._vectorizer.vocabulary_,
            "idf_": self._vectorizer.idf_.tolist(),
        }
        with open(str(path), "w", encoding="utf-8") as file:
            json.dump(data_to_save, file, ensure_ascii=False, indent=4)

    def load_vectorizer(self, filename: str) -> None:
        path = self._artifacts_dir_path / filename
        with open(str(path), "r", encoding="utf-8") as file:
            data_to_load = json.load(file)
        self._vectorizer = TfidfVectorizer(vocabulary=data_to_load["vocabulary_"])
        self._vectorizer.idf_ = np.array(data_to_load["idf_"])
