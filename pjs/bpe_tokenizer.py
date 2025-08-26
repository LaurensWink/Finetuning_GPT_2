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
from transformers import PreTrainedTokenizerFast


def gen_bpe_de_tokenizer(train_data, save_dir):
    special_tokens = ["PAD", "UNK", "UTT_BOUNDARY"]
    
    tokenizer = Tokenizer(models.BPE(unk_token="UNK"))
    tokenizer.normalizer = normalizers.Sequence([
        Strip(), NFC(), Lowercase()
    ])
    
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=16000,
        special_tokens=special_tokens
    )
    
    with open(train_data, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    tokenizer.train_from_iterator(lines, trainer=trainer)

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="UNK",
        pad_token="PAD",
        bos_token="UTT_BOUNDARY",
        eos_token="UTT_BOUNDARY"
    )
    
    os.makedirs(save_dir, exist_ok=True)
    wrapped_tokenizer.save_pretrained(save_dir)


# Trainieren & speichern
gen_bpe_de_tokenizer("data/baby_LM/trimmed_babylm_de.txt", "data/tokenizer/bpe_de_tokenizer")
gen_bpe_de_tokenizer("data/baby_LM/trimmed_babylm_en.txt", "data/tokenizer/bpe_en_tokenizer")

# Laden & testen
# my_bpe_tokenizer = PreTrainedTokenizerFast.from_pretrained("data/tokenizer/bpe_de_tokenizer")

# text = "Überraschung! Das ist überraschend"
# out = my_bpe_tokenizer(text, add_special_tokens=True)

# print("=== Input IDs & Tokens ===")
# for idx in out["input_ids"]:
#     token = my_bpe_tokenizer.convert_ids_to_tokens(idx)
#     print(f"{idx:5}  {token!r}  ->  {my_bpe_tokenizer.decode([idx])!r}")

# # Ganze dekodierte Sequenz
# decoded_text = my_bpe_tokenizer.decode(out["input_ids"])
# print("\n=== Decoded Text ===")
# print(decoded_text)

