import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from config import TrainMainConfig


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainMainConfig)


@hydra.main(config_path="configs", config_name="train_conf", version_base=None)
def main(cfg: TrainMainConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
