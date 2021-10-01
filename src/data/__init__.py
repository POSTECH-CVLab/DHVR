import gin
from src.data.collate import CollationFunctionFactory
from src.data.inf_sampler import InfSampler
from src.data.threedmatch_loader import *

ALL_DATASETS = [
    ThreeDMatchPairDataset07,
    ThreeDMatchPairDataset05,
    ThreeDMatchPairDataset03,
    ThreeDMatchTestDataset,
    ThreeDLoMatchTestDataset,
]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


@gin.configurable()
def get_dataset(dataset_name: str):
    if dataset_name not in dataset_str_mapping.keys():
        logging.error(
            f"Dataset {dataset_name}, does not exists in ".join(
                dataset_str_mapping.keys()
            )
        )

    return dataset_str_mapping[dataset_name]


def make_data_loader(
    dataset,
    phase="train",
    batch_size=1,
    num_workers=0,
    shuffle=None,
    collation_type="collate_pair",
):
    assert phase in ["train", "trainval", "val", "test"]
    if shuffle is None:
        shuffle = phase != "test"

    collation_fn = CollationFunctionFactory(collation_type)

    loader_dict = dict(
        batch_size=batch_size, collate_fn=collation_fn, num_workers=num_workers
    )
    if phase != "test":
        loader_dict["sampler"] = InfSampler(dataset, shuffle)

    loader = torch.utils.data.DataLoader(dataset, **loader_dict)

    return loader
