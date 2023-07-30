from datasets import load_dataset
from vinucmer.const import RAW_DATASET


def get_raw_dataset():
    dataset = load_dataset(RAW_DATASET, split='train')
    return dataset

