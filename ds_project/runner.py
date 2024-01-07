from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .dataset import NewsDataset
from .models import NewsClassificationModel
from .vectorization import NewsVectorizer


class TrainRunner:
    def __init__(
        self,
        dataset: NewsDataset,
        model: NewsClassificationModel,
        vectorizer: NewsVectorizer,
        vectorizer_saving_filename: str,
        model_saving_filename: str,
    ) -> None:
        self.dataset = dataset
        self.train_dataset = None
        self.eval_dataset = None
        self.eval_predictions = None
        self.model = model
        self.vectorizer = vectorizer
        self.vectorizer_saving_filename = vectorizer_saving_filename
        self.model_saving_filename = model_saving_filename

    def _split_to_train_eval(self):
        x_train, x_eval, y_train, y_eval = train_test_split(
            self.dataset.data,
            self.dataset.targets,
            random_state=42,
        )
        self.train_dataset = NewsDataset(
            data=x_train,
            targets=y_train,
            target_names=self.dataset.target_names,
        )
        self.eval_dataset = NewsDataset(
            data=x_eval,
            targets=y_eval,
            target_names=self.dataset.target_names,
        )
        return x_train, x_eval, y_train, y_eval

    def _save_vectorizer_data(self):
        self.vectorizer.save_vectorizer(self.vectorizer_saving_filename)

    def _save_model_data(self):
        self.model.save_model(self.model_saving_filename)

    def run(self):
        self.vectorizer.fit(self.dataset.data)
        self._split_to_train_eval()
        self.model.fit(
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )
        self.model.log_metrics()
        self.model.log_model("catboost_model")
        self._save_vectorizer_data()
        self._save_model_data()


class InferRunner:
    def __init__(
        self,
        dataset: NewsDataset,
        model: NewsClassificationModel,
        vectorizer: NewsVectorizer,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.vectorizer = vectorizer

    def _save_prediction_to_csv(self, prediction: np.ndarray, filename: str) -> None:
        path = Path("predictions")
        path.mkdir(parents=True, exist_ok=True)
        prediction.tofile(str(path / filename), sep=",")

    def run(self) -> None:
        prediction = self.model.predict(data=self.dataset.data)
        self._save_prediction_to_csv(prediction, "prediction.csv")
