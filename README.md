# Viral Nucleotide Transformers
## Quickstart
This is a guide to training a fine-tuned model using the Kaggle competition [Stanford Ribonanza RNA Folding](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding) dataset. 
For pre-training the model, the [NCBI Virus Complete DNA](https://huggingface.co/datasets/LKarlo/ncbi-virus-complete-dna-v230722) dataset is used.

### Pre-requisites
- Kaggle API in your local machine
- Install requirements

### Downloading the dataset
Download a version of the competition dataset that contains only a portion of the competition dataset. The dataset can be found in the [Huggingface dataset repository](https://huggingface.co/datasets/LKarlo/ncbi-virus-complete-dna-v230722).
```bash
kaggle datasets download -d leekh7411/stanford-ribonanza-rna-folding-light-version
```

### Corpus generation
Generate a corpus for nucleotide transformer pre-training.

The sequence corpus is generated using DNA, RNA virus nucleotide sequence data. The dataset can be found in the [Huggingface dataset repository](https://huggingface.co/datasets/LKarlo/ncbi-virus-complete-dna-v230722). This code repository contains code to download the dataset. The following command is the command to download the dataset and generate the corpus.

The length of the sequence in the dataset is very short, so the length of the corpus is not generated too long. 
The corpus length is set to 100 ~ 1000 bp and the number of corpus samples is set to 50,000.
```bash
python -m vinucmer corpus-generator -s temp/kaggle/data/corpus -sp ' ' -min 100 -max 1000 -min_s 1000 -max_s 10000 -n 50000
```

### Pre-corpus generation to train tokenizer
Use the same data source as the corpus created earlier. Extract the pre-corpus for tokenizer training from the data source. The extracted pre-corpus is used for tokenizer training and is saved in a format for this purpose. Use the following command to extract the pre-corpus.
```bash
python -m vinucmer pre-corpus -f temp/kaggle/data/pre-corpus.txt -s 6 -o 3 -r 30000 -s 42
```


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