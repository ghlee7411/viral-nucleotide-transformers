from vinucmers.pre_corpus import main as pre_corpus_main
from vinucmers.bpe_tokenizer import train as bpe_tokenizer_train
import click


@click.group()
def _pre_corpus():
    pass


@_pre_corpus.command()
@click.option('--file_path', '-f', type=str, required=True, help='Path to file')
@click.option('--split_size', '-s', type=int, default=10, help='Size of token')
@click.option('--overlap_size', '-o', type=int, default=5, help='Size of overlap')
@click.option('--random_sample_size', '-r', type=int, default=100000, help='Size of random sample')
@click.option('--seed', '-s', type=int, default=42, help='Random seed')
def pre_corpus(file_path: str, split_size: int=10, overlap_size: int=5, random_sample_size: int=100000, seed: int=42):
    """Command on pre-corpus to train nucleotide tokenizers"""
    pre_corpus_main(file_path, split_size, overlap_size, random_sample_size, seed)


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
    """Command on bpe_tokenizer"""
    bpe_tokenizer_train(pre_corpus_file, vocab_size, min_frequency, save_path, seed, unk_token, sep_token, mask_token, cls_token, pad_token)
    pass


cli = click.CommandCollection(sources=[_pre_corpus, _train_bpe_tokenizer])


if __name__ == '__main__':
    cli()