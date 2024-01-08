import logging

import hydra
import mlflow
from hydra.core.config_store import ConfigStore

from config import MainConfig
from news_classification.dataset import get_dataset
from news_classification.models import NewsClassificationModel
from news_classification.runner import TrainRunner
from news_classification.vectorization import NewsVectorizer


logger = logging.getLogger(__name__)


cs = ConfigStore.instance()
cs.store(name="main_config", node=MainConfig)


@hydra.main(config_path="configs", config_name="main_conf", version_base=None)
def main(cfg: MainConfig) -> None:
    mlflow.set_tracking_uri(uri=cfg.urls.mlflow_url)
    mlflow.set_experiment("News classification")
    logger.info("Loading data for model training")
    dataset = get_dataset(
        data_file=cfg.files.train_data,
        targets_file=cfg.files.train_targets,
        target_names_file=cfg.files.target_names,
    )
    vectorizer = NewsVectorizer()
    model = NewsClassificationModel(
        iterations=cfg.params.iterations,
        learning_rate=cfg.params.learning_rate,
        depth=cfg.params.tree_depth,
        vectorizer=vectorizer,
    )
    with mlflow.start_run():
        mlflow.log_params(
            {
                "iterations": cfg.params.iterations,
                "learning_rate": cfg.params.learning_rate,
                "depth": cfg.params.tree_depth,
            }
        )
        runner = TrainRunner(
            dataset=dataset,
            model=model,
            vectorizer=vectorizer,
            vectorizer_saving_filename=cfg.files.vectorizer,
            model_saving_filename=cfg.files.model,
        )
        logger.info("Starting model training")
        runner.run()
        logger.info("Model successfully trained")


if __name__ == "__main__":
    main()
