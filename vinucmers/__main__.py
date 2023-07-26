from vinucmers.pre_corpus import main as pre_corpus_main
from vinucmers.bpe_tokenizer import train as bpe_tokenizer_train
from vinucmers.unigram_tokenizer import train as unigram_tokenizer_train
from vinucmers.wordpiece_tokenizer import train as wordpiece_tokenizer_train
from vinucmers.models.roberta import train as roberta_train
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
    """ Download the raw sequence dataset(corpus) from Huggingface Dataset Hub and convert it to pre-corpus using window sliding technique."""
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
    """ Build a BPE tokenizer from pre-corpus using Huggingface Tokenizers. """
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
    """ Build a uni-gram tokenizer from pre-corpus using Huggingface Tokenizers. """
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
    """ Build a wordpiece tokenizer from pre-corpus using Huggingface Tokenizers. """
    wordpiece_tokenizer_train(
        pre_corpus_file, vocab_size, min_frequency, save_path, 
        seed, unk_token, sep_token, mask_token, cls_token, pad_token
    )


@click.group()
def _pretrain_roberta():
    pass


@_pretrain_roberta.command()
@click.option('--pre_tokenizer_path', '-p', type=str, required=True, help='Path to pre-tokenizer')
@click.option('--pretrained_save_path', '-s', type=str, required=True, help='Path to save pretrained model')
@click.option('--seed', type=int, default=42, help='Random seed')
def pretrain_roberta(
        pre_tokenizer_path: str,
        pretrained_save_path: str,
        seed: int=42
    ):
    """ Pretrain RoBERTa model. """
    roberta_train(pre_tokenizer_path, pretrained_save_path, seed)
    pass


cli = click.CommandCollection(sources=[_pre_corpus, _train_bpe_tokenizer, _train_unigram_tokenizer, _train_wordpiece_tokenizer, _pretrain_roberta])


if __name__ == '__main__':
    cli()