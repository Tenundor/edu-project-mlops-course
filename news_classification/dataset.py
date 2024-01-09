from .load_data import load_news_data, load_news_target_names, load_news_targets


class NewsDataset:
    def __init__(
        self, data: list[str], targets: list[int], target_names: list[str]
    ) -> None:
        if len(data) != len(targets):
            raise ValueError(
                "data and targets must be the same length. "
                f"{len(data)} != {len(targets)}"
            )
        self.data = data
        self.targets = targets
        self.target_names = target_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        news_data = self.get_news_data(idx)
        target = self.get_target(idx)
        return news_data, target

    def get_news_data(self, idx: int) -> str:
        return self.data[idx]

    def get_target(self, idx: int) -> int:
        return self.targets[idx]

    def get_target_name(self, idx: int) -> str:
        return self.target_names[idx]


def get_dataset(
    data_file: str,
    targets_file: str,
    target_names_file: str,
) -> NewsDataset:
    data = load_news_data(data_file)
    targets = load_news_targets(targets_file)
    target_names = load_news_target_names(target_names_file)
    return NewsDataset(
        data=data,
        targets=targets,
        target_names=target_names,
    )
