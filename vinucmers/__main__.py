from vinucmers.pre_corpus import main as pre_corpus_main
import click


@click.group()
def _pre_corpus():
    pass

@_pre_corpus.command()
@click.option('--file_path', '-f', type=str, required=True, help='Path to file')
@click.option('--split_size', '-s', type=int, default=10, help='Size of token')
@click.option('--overlap_size', '-o', type=int, default=5, help='Size of overlap')
def pre_corpus(file_path: str, split_size: int=10, overlap_size: int=5):
    """Command on pre-corpus to train nucleotide tokenizers"""
    pre_corpus_main(file_path, split_size, overlap_size)

cli = click.CommandCollection(sources=[_pre_corpus])


if __name__ == '__main__':
    cli()