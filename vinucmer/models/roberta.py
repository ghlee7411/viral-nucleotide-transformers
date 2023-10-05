from tokenizers import Tokenizer
from tokenizers.processors import RobertaProcessing
from tokenizers import models
from transformers import RobertaConfig, PreTrainedTokenizerFast, RobertaForMaskedLM
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from datasets import load_dataset
from vinucmer.utils import create_logger, sample_substring
from vinucmer.dataset import get_raw_dataset
import os


def train(
        pre_tokenizer_path: str,
        corpus_dir: str,
        pretrained_save_path: str,
        seed: int=42
    ):
    logger = create_logger(__name__)
    logger.info('Training viral nucleotide transformer model - RoBERTa')

    # Constants
    MAX_LENGTH = 512
    TRAIN_TEST_SPLIT = 0.05
    MIN_SUB_SEQ_LENGTH = 50
    MAX_SUB_SEQ_LENGTH = 2000
    MLM_PROBABILITY = 0.15
    NUM_PROCESS = 10

    # Create pretrained_save_path if not exists
    if not os.path.exists(pretrained_save_path):
        logger.info(f'Creating {pretrained_save_path} directory')
        os.makedirs(pretrained_save_path)

    # Setup tokenizer for Roberta
    pre_tokenizer = Tokenizer.from_file(pre_tokenizer_path)
    pre_tokenizer.post_processor = RobertaProcessing(
        ("<SEP>", pre_tokenizer.token_to_id("<SEP>")),
        ("<CLS>", pre_tokenizer.token_to_id("<CLS>")),
    )
    _pre_tokenizer_path = os.path.join(pretrained_save_path, 'pre_tokenizer.json')
    pre_tokenizer.enable_truncation(max_length=MAX_LENGTH)
    pre_tokenizer.save(_pre_tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=_pre_tokenizer_path)
    tokenizer.add_special_tokens({
        'unk_token': "<UNK>", 
        'sep_token': "<SEP>", 
        'mask_token': "<MASK>", 
        'cls_token': "<CLS>",
        'pad_token': "<PAD>"
    })
    tokenizer.save_pretrained(pretrained_save_path)

    # Reload tokenizer from pretrained_save_path
    tokenizer = AutoTokenizer.from_pretrained(pretrained_save_path, model_max_length=MAX_LENGTH)
    tokenizer.save_pretrained(pretrained_save_path)

    # Load dataset
    logger.info('Loading dataset..')

    def sample_subsequence(examples):
        sub_seqs = list()
        for s in examples['text']:
            subs = sample_substring(s, MIN_SUB_SEQ_LENGTH, MAX_SUB_SEQ_LENGTH)
            if subs:
                sub_seqs.append(subs)

        return tokenizer(sub_seqs, padding='max_length', truncation=True)

    dataset = load_dataset('text', data_dir=corpus_dir, split='train')
    
    dataset = dataset.map(sample_subsequence, batched=True, num_proc=NUM_PROCESS)
    dataset = dataset.train_test_split(test_size=TRAIN_TEST_SPLIT, seed=seed, shuffle=True)

    print(dataset)
    print(dataset['train'][0])

    # Setup config for Roberta
    config = RobertaConfig(vocab_size=len(tokenizer))
    model = RobertaForMaskedLM(config=config)

    # Masked language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROBABILITY
    )

    model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")
    training_args = TrainingArguments(
        output_dir=pretrained_save_path,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
        report_to="none",
        use_mps_device=True # Apple M1 GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )

    trainer.train()

    return