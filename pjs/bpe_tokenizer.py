import os
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer)
from tokenizers.normalizers import Lowercase, Strip, NFC, Sequence
from transformers import LlamaTokenizerFast

def gen_bpe_de_tokenizer(train_data, save_dir):
    special_tokens = ["PAD", "UNK", "UTT_BOUNDARY"]
    
    tokenizer = Tokenizer(models.BPE(unk_token="UNK"))
    tokenizer.normalizer = Sequence([
        NFC(),
        Lowercase(),
        Strip()
    ])
    
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=16000,
        special_tokens=special_tokens,
    )
    
    tokenizer.train(files = train_data, trainer=trainer)
    pad_id = tokenizer.token_to_id("PAD")
    tokenizer.enable_padding(pad_id=pad_id, pad_token="PAD")
    tokenizer.enable_truncation(max_length=256)
    
    wrapped_tokenizer = LlamaTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="PAD",
        bos_token="UTT_BOUNDARY",
        eos_token="UTT_BOUNDARY",
        unk_token="UNK",
        legacy=False
    )

    wrapped_tokenizer.model_max_length = 256
    
    os.makedirs(save_dir, exist_ok=True)
    wrapped_tokenizer.save_pretrained(save_dir)


gen_bpe_de_tokenizer(["data/baby_LM/trimmed_babylm_de.txt"], "data/tokenizer/bpe_de_tokenizer")
gen_bpe_de_tokenizer(["data/baby_LM/trimmed_babylm_en.txt"], "data/tokenizer/bpe_en_tokenizer")