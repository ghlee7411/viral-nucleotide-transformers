from vinucmer.utils import create_logger
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import models
from tokenizers import Tokenizer


def train(
        pre_corpus_file: str, 
        vocab_size: int=10000, 
        min_frequency: int=2, 
        save_path: str='bpe_tokenizer.json',
        seed: int=42,
        unk_token: str="<UNK>",
        sep_token: str="<SEP>",
        mask_token: str="<MASK>",
        cls_token: str="<CLS>",
        pad_token: str="<PAD>"
    ):
    """ Main function to train bpe tokenizer.

    Args:
        pre_corpus_file: path to pre-corpus file

    Returns:
        None
    """
    logger = create_logger(__name__)
    logger.info('Training bpe tokenizer')
    model = models.BPE(unk_token=unk_token)
    tokenizer = Tokenizer(model=model)
    trainer = BpeTrainer(
        vocab_size=vocab_size, 
        min_frequency=min_frequency, 
        special_tokens=[unk_token, sep_token, mask_token, cls_token, pad_token], 
        show_progress=True,
        seed=seed
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(files=[pre_corpus_file], trainer=trainer)
    tokenizer.save(save_path)
    return