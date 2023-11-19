from vinucmer.pre_corpus import main as pre_corpus_main
from vinucmer.bpe_tokenizer import train as bpe_tokenizer_train
from vinucmer.unigram_tokenizer import train as unigram_tokenizer_train
from vinucmer.wordpiece_tokenizer import train as wordpiece_tokenizer_train
from vinucmer.models.roberta import train as roberta_train
from vinucmer.corpus_generator import main as corpus_main
import click


@click.group()
def _pre_corpus():
    pass


@_pre_corpus.command()
@click.option('--file_path', '-f', type=str, required=True, help='Path to file')
@click.option('--token_size', '-t', type=int, default=10, help='Size of token')
@click.option('--overlap_size', '-o', type=int, default=5, help='Size of overlap')
@click.option('--random_sample_size', '-r', type=int, default=100000, help='Size of random sample')
@click.option('--seed', '-s', type=int, default=42, help='Random seed')
def pre_corpus(file_path: str, token_size: int=10, overlap_size: int=5, random_sample_size: int=100000, seed: int=42):
    """ Preprocess viral nucleotide sequence data for tokenization."""
    pre_corpus_main(file_path, token_size, overlap_size, random_sample_size, seed)


@click.group()
def _train_bpe_tokenizer():
    pass


@_train_bpe_tokenizer.command()
@click.option('--pre_corpus_file', '-f', type=str, required=True, help='Path to pre-corpus file')
@click.option('--vocab_size', '-v', type=int, default=10000, help='Vocab size')
@click.option('--min_frequency', '-m', type=int, default=2, help='Min frequency')
@click.option('--save_path', '-s', type=str, default='bpe_tokenizer.json', help='Path to save bpe tokenizer')
@click.option('--seed', type=int, default=42, help='Random seed')
@click.option('--unk_token', '-u', type=str, default="<UNK>", help='UNK token')
@click.option('--sep_token', '-e', type=str, default="<SEP>", help='SEP token')
@click.option('--mask_token', '-k', type=str, default="<MASK>", help='MASK token')
@click.option('--cls_token', '-c', type=str, default="<CLS>", help='CLS token')
@click.option('--pad_token', '-p', type=str, default="<PAD>", help='PAD token')
def train_bpe_tokenizer(
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
    """ Build a BPE tokenizer from the <pre-corpus> using Huggingface Tokenizers. """
    bpe_tokenizer_train(
        pre_corpus_file, vocab_size, min_frequency, save_path, 
        seed, unk_token, sep_token, mask_token, cls_token, pad_token
    )


@click.group()
def _train_unigram_tokenizer():
    pass


@_train_unigram_tokenizer.command()
@click.option('--pre_corpus_file', '-f', type=str, required=True, help='Path to pre-corpus file')
@click.option('--vocab_size', '-v', type=int, default=10000, help='Vocab size')
@click.option('--shrink_factor', '-sf', type=float, default=0.75, help='Shrink factor')
@click.option('--max_piece_length', '-m', type=int, default=16, help='Max piece length')
@click.option('--n_sub_iterations', '-n', type=int, default=2, help='Number of sub iterations')
@click.option('--save_path', '-s', type=str, default='unigram_tokenizer.json', help='Path to save unigram tokenizer')
@click.option('--seed', type=int, default=42, help='Random seed')
@click.option('--unk_token', '-u', type=str, default="<UNK>", help='UNK token')
@click.option('--sep_token', '-e', type=str, default="<SEP>", help='SEP token')
@click.option('--mask_token', '-k', type=str, default="<MASK>", help='MASK token')
@click.option('--cls_token', '-c', type=str, default="<CLS>", help='CLS token')
@click.option('--pad_token', '-p', type=str, default="<PAD>", help='PAD token')
def train_unigram_tokenizer(
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
    """ Build a uni-gram tokenizer from the <pre-corpus> using Huggingface Tokenizers. """
    unigram_tokenizer_train(
        pre_corpus_file, vocab_size, shrink_factor, max_piece_length, n_sub_iterations, 
        save_path, seed, unk_token, sep_token, mask_token, cls_token, pad_token
    )


@click.group()
def _train_wordpiece_tokenizer():
    pass


@_train_wordpiece_tokenizer.command()
@click.option('--pre_corpus_file', '-f', type=str, required=True, help='Path to pre-corpus file')
@click.option('--vocab_size', '-v', type=int, default=10000, help='Vocab size')
@click.option('--min_frequency', '-m', type=int, default=2, help='Min frequency')
@click.option('--save_path', '-s', type=str, default='wordpiece_tokenizer.json', help='Path to save wordpiece tokenizer')
@click.option('--seed', type=int, default=42, help='Random seed')
@click.option('--unk_token', '-u', type=str, default="<UNK>", help='UNK token')
@click.option('--sep_token', '-e', type=str, default="<SEP>", help='SEP token')
@click.option('--mask_token', '-k', type=str, default="<MASK>", help='MASK token')
@click.option('--cls_token', '-c', type=str, default="<CLS>", help='CLS token')
@click.option('--pad_token', '-p', type=str, default="<PAD>", help='PAD token')
def train_wordpiece_tokenizer(
        pre_corpus_file: str, 
        vocab_size: int=10000, 
        min_frequency: int=2,
        save_path: str='wordpiece_tokenizer.json',
        seed: int=42,
        unk_token: str="<UNK>",
        sep_token: str="<SEP>",
        mask_token: str="<MASK>",
        cls_token: str="<CLS>",
        pad_token: str="<PAD>"
    ):
    """ Build a wordpiece tokenizer from the <pre-corpus> using Huggingface Tokenizers. """
    wordpiece_tokenizer_train(
        pre_corpus_file, vocab_size, min_frequency, save_path, 
        seed, unk_token, sep_token, mask_token, cls_token, pad_token
    )


@click.group()
def _pretrain_roberta():
    pass


@_pretrain_roberta.command()
@click.option('--pre_tokenizer_path', '-p', type=str, required=True, help='Path to pre-tokenizer')
@click.option('--corpus_dir', '-c', type=str, required=True, help='Path to corpus directory')
@click.option('--pretrained_save_path', '-s', type=str, required=True, help='Path to save pretrained model')
@click.option('--repo_id', '-r', type=str, required=True, help='Huggingface Model repository id to push model weights after training')
@click.option('--seed', type=int, default=42, help='Random seed')
def pretrain_roberta(
        pre_tokenizer_path: str,
        corpus_dir: str,
        pretrained_save_path: str,
        repo_id: str,
        seed: int=42
    ):
    """ Pretrain RoBERTa model. """
    roberta_train(pre_tokenizer_path, corpus_dir, pretrained_save_path, repo_id, seed)
    pass


@click.group()
def _corpus_generator():
    pass


@_corpus_generator.command()
@click.option('--save_dir', '-s', type=str, required=True, help='Path to save directory')
@click.option('--splitter', '-sp', type=str, default=None, help='Splitter')
@click.option('--min_bp', '-min', type=int, default=30, help='Min bp')
@click.option('--max_bp', '-max', type=int, default=2000, help='Max bp')
@click.option('--min_samples', '-min_s', type=int, default=5, help='Min samples')
@click.option('--max_samples', '-max_s', type=int, default=1000, help='Max samples')
@click.option('--seed', type=int, default=42, help='Random seed')
@click.option('--num_sample_raw', '-n', type=int, default=None, help='Number of raw samples')
def corpus_generator(save_dir: str, splitter: str = None, min_bp: int = 30, max_bp: int = 2000, min_samples: int = 5, max_samples: int = 1000, seed: int = 42, num_sample_raw: int = None):
    """ Generate corpus for nucleotide transformer pre-training."""
    corpus_main(
        save_dir, splitter, min_bp, max_bp, min_samples, max_samples, seed, num_sample_raw
    )


cli = click.CommandCollection(
    sources=[
        _pre_corpus, 
        _corpus_generator,
        _train_bpe_tokenizer, 
        _train_unigram_tokenizer, 
        _train_wordpiece_tokenizer, 
        _pretrain_roberta
    ]
)


if __name__ == '__main__':
    cli()