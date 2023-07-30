# Docstring for sphinx
from vinucmer.utils import create_logger
from vinucmer.dataset import get_raw_dataset
import os
import random
import shutil


def sample_subsequence(sequence, subseq_length):
    if len(sequence) <= subseq_length:
        raise ValueError("Subsequence length should be smaller than the sequence length.")
    
    start_index = random.randint(0, len(sequence) - subseq_length)
    sampled_subseq = sequence[start_index:start_index + subseq_length]
    return sampled_subseq


def get_subsequences(sequence: str, min_bp: int = 30, max_bp: int = 2000, min_samples: int = 5, max_samples: int = 1000, seed: int = 42):
    """ Get subsequences from a sequence.
    
    Args:
        sequence (str): sequence
        min_bp (int): minimum base pair length, default 30
        max_bp (int): maximum base pair length, default 2000
        min_samples (int): minimum number of samples, default 5
        max_samples (int): maximum number of samples, default 1000
        seed (int): random seed, default 42

    Returns:
        number of subsequences
    """
    subsequences = list()
    random.seed(seed)
    num_samples = int((len(sequence) // max_bp))
    num_samples = max(num_samples, min_samples)
    num_samples = min(num_samples, max_samples)
    for _ in range(num_samples):
        subseq_length = random.randint(min_bp, max_bp)
        subseq_length = min(subseq_length, len(sequence)-1)
        subseq = sample_subsequence(sequence, subseq_length)
        subsequences.append(subseq)
    return subsequences


def main(save_dir: str, splitter: str = None, min_bp: int = 30, max_bp: int = 2000, min_samples: int = 5, max_samples: int = 1000, seed: int = 42, num_sample_raw: int = None):
    """ Main function to run nucleotide corpus generator.
    
    Args:
        save_dir (str): path to save directory
        splitter (str): splitter, default None
        min_bp (int): minimum base pair length, default 30
        max_bp (int): maximum base pair length, default 2000
        min_samples (int): minimum number of samples, default 5
        max_samples (int): maximum number of samples, default 1000
        seed (int): random seed, default 42
        num_sample_raw (int): number of raw samples, default None

    Returns:
        None
    """
    logger = create_logger(__name__)
    logger.info('Making nucleotide corpus to pre-train nucleotide transformers')

    # logger check input parameters
    logger.info('[Input parameters]')
    logger.info(f'save_dir: {save_dir}')
    logger.info(f'splitter: {splitter}')
    logger.info(f'min_bp: {min_bp}')
    logger.info(f'max_bp: {max_bp}')
    logger.info(f'min_samples: {min_samples}')
    logger.info(f'max_samples: {max_samples}')
    logger.info(f'seed: {seed}')
    logger.info(f'num_sample_raw: {num_sample_raw}')

    if os.path.exists(save_dir):
        logger.info(f'Nucleotide corpus already exists in {save_dir}')
        # remove directory
        shutil.rmtree(save_dir)
        logger.info(f'Nucleotide corpus removed from {save_dir} successfully and will be recreated...')
        os.makedirs(save_dir)
    else:
        logger.info(f'Creating {save_dir} directory')
        os.makedirs(save_dir)

    dataset = get_raw_dataset()
    dataset = dataset.shuffle(seed=seed)
    default_save_path = os.path.join(save_dir, 'default.txt')
    num_iters = len(dataset)
    num_iters = num_sample_raw if num_sample_raw is not None else num_iters

    for i, data in enumerate(dataset):
        save_path = default_save_path
        split = 'default'
        if splitter is not None and splitter in data:
            split = data[splitter]
            save_path = os.path.join(save_dir, f'{split}.txt')
        
        nucleotide = data['Sequence']
        subsequences = get_subsequences(nucleotide, min_bp, max_bp, min_samples, max_samples, seed)
        
        if not os.path.exists(save_path):
            logger.info(f'Creating {save_path} file')
            with open(save_path, 'w') as f:
                f.write('')

        with open(save_path, 'a') as f:
            for subseq in subsequences:
                f.write(f'{subseq}\n')
        
        logger.info(f'[{i:>04}/{num_iters:>04}] Nucleotide corpus saved to {save_path} successfully with {len(subsequences)} subsequences')

        if num_sample_raw is not None and i >= num_sample_raw:
            logger.info(f'Number of raw samples reached {num_sample_raw}.')
            break
    

if __name__ == '__main__':
    main()