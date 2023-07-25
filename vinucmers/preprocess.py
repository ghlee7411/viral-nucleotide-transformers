# Docstring for sphinx
from vinucmers.utils import create_logger


def nucleotide_to_nuc_words(nucleotide: str, split_size: int=10, overlap_size: int=5):
    """
    Converts nucleotide sequence to pre-words.

    :param nucleotide: nucleotide sequence
    :param split_size: size of token
    :param overlap_size: size of overlap
    :return: list of tokens
    """
    tokens = []
    for i in range(0, len(nucleotide), split_size - overlap_size):
        token = nucleotide[i:i + split_size]
        if len(token) == split_size:
            tokens.append(token)
    return tokens


def nuc_words_to_nucleotide(tokens: list, split_size: int=10, overlap_size: int=5):
    """
    Converts nucleotide pre-words to nucleotide sequence.

    :param tokens: list of tokens
    :param split_size: size of token
    :param overlap_size: size of overlap
    :return: nucleotide sequence
    """
    nucleotide = ''
    for i, token in enumerate(tokens):
        nucleotide += token
        if i != len(tokens) - 1:
            nucleotide = nucleotide[:-overlap_size]
    return nucleotide


def append_nuc_words_to_file(file_path: str, tokens: list):
    """
    Appends tokens to file.

    :param file_path: path to file
    :param tokens: list of tokens
    """
    with open(file_path, 'a') as f:
        for token in tokens:
            f.write(token + '\n')
