import yaml
import json

class SkipDataLoader(DataLoader):
        """
        Subclass of a PyTorch `DataLoader` that will skip the first batches.

        Args:
            dataset (`torch.utils.data.dataset.Dataset`):
                The dataset to use to build this datalaoder.
            skip_batches (`int`, *optional*, defaults to 0):
                The number of batches to skip at the beginning.
            kwargs:
                All other keyword arguments to pass to the regular `DataLoader` initialization.
        """

        def __init__(self, dataset, skip_batches=0, **kwargs):
            super().__init__(dataset, **kwargs)
            self.skip_batches = skip_batches

        def __iter__(self):
            for index, batch in enumerate(tqdm(super().__iter__())):
                if index >= self.skip_batches:
                    yield batch

def load_yaml(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def load_json(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data 