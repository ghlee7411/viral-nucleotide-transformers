# Docstring for sphinx
from vinucmer.utils import create_logger
from vinucmer.dataset import get_raw_dataset
import os


def nucleotide_to_nuc_words(nucleotide: str, split_size: int=10, overlap_size: int=5):
    """ Converts nucleotide sequence to pre-words.

    Args:
        nucleotide: nucleotide sequence
        split_size: size of token
        overlap_size: size of overlap

    Returns:
        list of tokens
    """
    tokens = []
    for i in range(0, len(nucleotide), split_size - overlap_size):
        token = nucleotide[i:i + split_size]
        if len(token) == split_size:
            tokens.append(token)
    return tokens


def nuc_words_to_nucleotide(tokens: list, split_size: int=10, overlap_size: int=5):
    """ Converts pre-words to nucleotide sequence.

    Args:
        tokens: list of tokens
        split_size: size of token
        overlap_size: size of overlap

    Returns:
        nucleotide sequence
    """
    nucleotide = ''
    for i, token in enumerate(tokens):
        nucleotide += token
        if i != len(tokens) - 1:
            nucleotide = nucleotide[:-overlap_size]
    return nucleotide


def append_nuc_words_to_file(file_path: str, tokens: list):
    """ Appends tokens to file.

    Args:
        file_path: path to file
        tokens: list of tokens

    Returns:
        None
    """
    with open(file_path, 'a') as f:
        split_nucleotide = ' '.join(tokens)
        f.write(split_nucleotide + '\n')


def main(file_path: str, split_size: int=10, overlap_size: int=5, random_sample_size: int=100000, seed: int=42):
    """ Main function to run pre-corpus.

    Args:
        file_path (str): path to file
        split_size (int): size of token, default 10
        overlap_size (int): size of overlap, default 5
        random_sample_size (int): size of random sample, default 100000
        seed (int): random seed, default 42

    Returns:
        None
    """
    logger = create_logger(__name__)
    logger.info('Making pre-corpus to train nucleotide tokenizers')
    logger.info(f'File path: {file_path}')
    logger.info(f'Split size: {split_size}')
    logger.info(f'Overlap size: {overlap_size}')
    logger.info(f'Random sample size: {random_sample_size}')
    logger.info(f'Seed: {seed}')

    if os.path.exists(file_path):
        logger.info(f'Pre-corpus already exists in {file_path}')
        # remove file
        os.remove(file_path)
        logger.info(f'Pre-corpus removed from {file_path} successfully and will be recreated...')

    dataset = get_raw_dataset()

    if random_sample_size > len(dataset):
        dataset = dataset.shuffle(seed=seed)
    
    num_samples = min(random_sample_size, len(dataset))

    for i, data in enumerate(dataset):
        nucleotide = data['Sequence']
        tokens = nucleotide_to_nuc_words(nucleotide, split_size, overlap_size)
        append_nuc_words_to_file(file_path, tokens)
        
        if i % 1000 == 0:
            logger.info(f'{i}/{num_samples} sequences processed')

        if i == num_samples:
            logger.info(f'Pre-corpus created with {i} sequences randomly sampled from {len(dataset)} sequences')
            break
    
    logger.info(f'Pre-corpus created successfully in {file_path} with size {i} sequences')


if __name__ == '__main__':
    main()