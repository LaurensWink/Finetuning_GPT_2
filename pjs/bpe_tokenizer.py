import os
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer)

from tokenizers.normalizers import Lowercase, Strip, NFC

from transformers import (
    AutoTokenizer, 
    PreTrainedTokenizerFast, 
    set_seed, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    LlamaForCausalLM, 
    LlamaConfig)

from datasets import load_dataset

def gen_bpe_de_tokenizer():
    special_tokens = ["PAD", "UNK", "UTT_BOUNDARY"]
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence([
        NFC(), Lowercase(), Strip()
    ])
    
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    trainer = trainers.BpeTrainer(
        vocab_size=16000,
        special_tokens=special_tokens
    )
    
    tokenizer.train(files = ['data/baby_LM/trimmed_babylm_de.txt'], trainer=trainer)
    
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="UNK",
        pad_token="PAD",
        bos_token="UTT_BOUNDARY",
        eos_token="UTT_BOUNDARY"
    )
    
    os.makedirs("data/tokenizer/bpe_de_tokenizer", exist_ok=True)
    wrapped_tokenizer.save_pretrained("data/tokenizer/bpe_de_tokenizer")

gen_bpe_de_tokenizer()

my_bpe_tokenizer = PreTrainedTokenizerFast.from_pretrained("data/tokenizer/bpe_de_tokenizer")
tokenizer = AutoTokenizer.from_pretrained('phonemetransformers/babble-tokenizers', subfolder='BABYLM-TOKENIZER-BPE-TXT')

text = "Überraschung! Das ist überraschend"

my_output = my_bpe_tokenizer(text, add_special_tokens=True)
ext_output = tokenizer(text, add_special_tokens=True)

# IDs ausgeben
print("=== Eigener Char-Tokenizer ===")
print("Input IDs:", my_output["input_ids"])
print("Tokens:   ", my_bpe_tokenizer.convert_ids_to_tokens(my_output["input_ids"]))

print("\n=== Babble Tokenizer ===")
print("Input IDs:", ext_output["input_ids"])
print("Tokens:   ", tokenizer.convert_ids_to_tokens(ext_output["input_ids"]))

# Optional: Vergleich der Token-Längen
print("\nTokenanzahl Eigener Tokenizer:", len(my_output["input_ids"]))
print("Tokenanzahl Babble Tokenizer:", len(ext_output["input_ids"]))
