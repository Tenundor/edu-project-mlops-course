from pathlib import Path

import mlflow
from catboost import CatBoostClassifier, Pool

from .dataset import NewsDataset
from .exceptions import ModelException
from .vectorization import NewsVectorizer


class NewsClassificationModel:
    def __init__(
        self,
        vectorizer: NewsVectorizer,
        iterations: int | None = None,
        learning_rate: float | None = None,
        depth: int | None = None,
    ):
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            task_type="CPU",
            verbose=False,
            loss_function="MultiClass",
            custom_metric=["Accuracy", "AUC"],
        )
        self._vectorizer = vectorizer
        self._model_dir_path = Path("models")
        self.evals_result = None

    def fit(self, train_dataset: NewsDataset, eval_dataset: NewsDataset) -> None:
        train_vec = self._vectorizer.transform(train_dataset.data)
        eval_vec = self._vectorizer.transform(eval_dataset.data)
        train_pool = Pool(train_vec, label=train_dataset.targets)
        eval_pool = Pool(eval_vec, label=eval_dataset.targets)
        self.model.fit(train_pool, eval_set=eval_pool, use_best_model=True, plot=False)
        self.evals_result = self.model.get_evals_result()

    def predict(self, data: list[str]):
        data_vec = self._vectorizer.transform(data)
        return self.model.predict(data_vec)

    def save_model(self, filename: str) -> None:
        self._model_dir_path.mkdir(parents=True, exist_ok=True)
        path = self._model_dir_path / filename
        self.model.save_model(str(path), format="cbm")

    def load_model(self, filename: str) -> None:
        path = self._model_dir_path / filename
        self.model.load_model(str(path), format="cbm")

    @staticmethod
    def _log_metric(metric_name: str, data: list[float]):
        for step, value in enumerate(data):
            mlflow.log_metric(metric_name, value, step=step)

    def log_metrics(self):
        if self.evals_result is None:
            raise ModelException("There are no evals result for logging")
        learn_metrics = self.evals_result["learn"]
        validation_metrics = self.evals_result["validation"]
        self._log_metric("Accuracy_learn", learn_metrics["Accuracy"])
        self._log_metric("MultiClass_learn", learn_metrics["MultiClass"])
        self._log_metric("Accuracy_validation", validation_metrics["Accuracy"])
        self._log_metric("MultiClass_validation", validation_metrics["MultiClass"])
        self._log_metric("AUC", validation_metrics["AUC:type=Mu"])

    def log_model(self, log_name: str):
        mlflow.catboost.log_model(self.model, log_name)
