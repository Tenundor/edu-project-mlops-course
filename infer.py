import logging

import hydra
from hydra.core.config_store import ConfigStore

from config import MainConfig
from news_classification.dataset import get_dataset
from news_classification.models import NewsClassificationModel
from news_classification.runner import InferRunner
from news_classification.vectorization import NewsVectorizer


logger = logging.getLogger(__name__)


cs = ConfigStore.instance()
cs.store(name="main_config", node=MainConfig)


@hydra.main(config_path="configs", config_name="main_conf", version_base=None)
def main(cfg: MainConfig) -> None:
    dataset = get_dataset(
        data_file=cfg.files.test_data,
        targets_file=cfg.files.test_targets,
        target_names_file=cfg.files.target_names,
    )
    logger.info("Loading vectorizer")
    vectorizer = NewsVectorizer()
    vectorizer.load_vectorizer(filename=cfg.files.vectorizer)
    logger.info("Loading trained model")
    model = NewsClassificationModel(
        vectorizer=vectorizer,
    )
    model.load_model(filename=cfg.files.model)
    runner = InferRunner(
        dataset=dataset,
        model=model,
        vectorizer=vectorizer,
    )
    logger.info("Making prediction")
    runner.run()
    logger.info("Prediction saved successfully")


if __name__ == "__main__":
    main()
