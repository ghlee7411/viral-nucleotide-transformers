import dataset
from datasets import load_dataset
from vinucmers.const import RAW_DATASET


def get_raw_dataset():
    dataset = load_dataset(RAW_DATASET)
    return dataset

