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

python -m vinucmer corpus-generator \
--save_dir temp/kaggle/data/corpus \
--splitter ' ' \
--min_bp 100 \
--max_bp 1000 \
--min_samples 1000 \
--max_samples 10000 \
--num_sample_raw 10000

```

### Pre-corpus generation to train tokenizer
Use the same data source as the corpus created earlier. Extract the pre-corpus for tokenizer training from the data source. The extracted pre-corpus is used for tokenizer training and is saved in a format for this purpose. Use the following command to extract the pre-corpus.
```bash
python -m vinucmer pre-corpus -f temp/kaggle/data/pre-corpus.txt -t 4 -o 3 -r 30000 -s 42
```

### Tokenizer training
Train a tokenizer using the pre-corpus generated earlier. The tokenizer is trained using the Huggingface Tokenizers library. The following command is used to train the tokenizer.
```bash
python -m vinucmer train-bpe-tokenizer -f temp/kaggle/data/pre-corpus.txt -v 10000 -m 2 -s temp/kaggle/data/tokenizer.json --seed 42
```

### Pre-training
Pre-train the RoBERTa model using the tokenizer trained earlier and the corpus generated earlier. The following command is used to pre-train the model.
```bash
python -m vinucmer pretrain-roberta -p temp/kaggle/data/tokenizer.json -c temp/kaggle/data/corpus -s temp/kaggle/data/pretrained-model --seed 42
```