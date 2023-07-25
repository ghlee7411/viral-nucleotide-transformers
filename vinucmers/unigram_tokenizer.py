from vinucmers.utils import create_logger
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import models
from tokenizers import Tokenizer


def train(
        pre_corpus_file: str, 
        vocab_size: int=10000, 
        shrink_factor: float=0.75,
        max_piece_length: int=16,
        n_sub_iterations: int=2,
        save_path: str='unigram_tokenizer.json',
        seed: int=42,
        unk_token: str="<UNK>",
        sep_token: str="<SEP>",
        mask_token: str="<MASK>",
        cls_token: str="<CLS>",
        pad_token: str="<PAD>"
    ):
    """ Main function to train uni-gram tokenizer.

    Args:
        pre_corpus_file: path to pre-corpus file

    Returns:
        None
    """
    logger = create_logger(__name__)
    logger.info('Training unigram tokenizer')
    model = models.Unigram()
    tokenizer = Tokenizer(model=model)
    trainer = UnigramTrainer(
        vocab_size=vocab_size, 
        shrinking_factor=shrink_factor,
        max_piece_length=max_piece_length,
        n_sub_iterations=n_sub_iterations,
        special_tokens=[sep_token, mask_token, cls_token, pad_token], 
        unk_token=unk_token,
        show_progress=True,
        seed=seed
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(files=[pre_corpus_file], trainer=trainer)
    tokenizer.save(save_path)
    return