# Docstring for sphinx
from vinucmers.utils import create_logger
from vinucmers.dataset import get_raw_dataset
import click


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
        for token in tokens:
            f.write(token + '\n')


def main(file_path: str, split_size: int=10, overlap_size: int=5):
    """ Main function to run pre-corpus.

    Args:
        file_path: path to file
        split_size: size of token
        overlap_size: size of overlap

    Returns:
        None
    """
    logger = create_logger(__name__)
    logger.info('Making pre-corpus to train nucleotide tokenizers')
    dataset = get_raw_dataset()
    for i, data in enumerate(dataset):
        nucleotide = data['sequence']
        tokens = nucleotide_to_nuc_words(nucleotide, split_size, overlap_size)
        append_nuc_words_to_file(file_path, tokens)
        if i % 1000 == 0:
            logger.info(f'{i}/{len(dataset)} sequences processed')


if __name__ == '__main__':
    main()