# Viral Nucleotide Transformers
## Usage
```bash
# python -m vinucmer --help
Usage: python -m vinucmer [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  corpus-generator           Generate corpus for nucleotide transformer...
  pre-corpus                 Preprocess viral nucleotide sequence data...
  pretrain-roberta           Pretrain RoBERTa model.
  train-bpe-tokenizer        Build a BPE tokenizer from the <pre-corpus>...
  train-unigram-tokenizer    Build a uni-gram tokenizer from the...
  train-wordpiece-tokenizer  Build a wordpiece tokenizer from the...
```
```bash
# python -m vinucmer corpus-generator --help
Usage: python -m vinucmer corpus-generator [OPTIONS]

  Generate corpus for nucleotide transformer pre-training.

Options:
  -s, --save_dir TEXT            Path to save directory  [required]
  -sp, --splitter TEXT           Splitter
  -min, --min_bp INTEGER         Min bp
  -max, --max_bp INTEGER         Max bp
  -min_s, --min_samples INTEGER  Min samples
  -max_s, --max_samples INTEGER  Max samples
  --seed INTEGER                 Random seed
  -n, --num_sample_raw INTEGER   Number of raw samples
  --help                         Show this message and exit.
```
```bash
# python -m vinucmer pre-corpus --help
Usage: python -m vinucmer pre-corpus [OPTIONS]

  Preprocess viral nucleotide sequence data for tokenization.

Options:
  -f, --file_path TEXT            Path to file  [required]
  -s, --split_size INTEGER        Size of token
  -o, --overlap_size INTEGER      Size of overlap
  -r, --random_sample_size INTEGER
                                  Size of random sample
  -s, --seed INTEGER              Random seed
  --help                          Show this message and exit.
```
```bash
# python -m vinucmer train-bpe-tokenizer --help
Usage: python -m vinucmer train-bpe-tokenizer [OPTIONS]

  Build a BPE tokenizer from the <pre-corpus> using Huggingface Tokenizers.

Options:
  -f, --pre_corpus_file TEXT   Path to pre-corpus file  [required]
  -v, --vocab_size INTEGER     Vocab size
  -m, --min_frequency INTEGER  Min frequency
  -s, --save_path TEXT         Path to save bpe tokenizer
  --seed INTEGER               Random seed
  -u, --unk_token TEXT         UNK token
  -e, --sep_token TEXT         SEP token
  -k, --mask_token TEXT        MASK token
  -c, --cls_token TEXT         CLS token
  -p, --pad_token TEXT         PAD token
  --help                       Show this message and exit.
```
```bash
# python -m vinucmer pretrain-roberta --help
Usage: python -m vinucmer pretrain-roberta [OPTIONS]

  Pretrain RoBERTa model.

Options:
  -p, --pre_tokenizer_path TEXT   Path to pre-tokenizer  [required]
  -c, --corpus_dir TEXT           Path to corpus directory  [required]
  -s, --pretrained_save_path TEXT
                                  Path to save pretrained model  [required]
  --seed INTEGER                  Random seed
  --help                          Show this message and exit.
```