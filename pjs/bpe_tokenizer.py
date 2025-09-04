import os
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer)

from tokenizers.processors import TemplateProcessing

from tokenizers.normalizers import Lowercase, Strip, NFC
from transformers import PreTrainedTokenizerFast

def gen_bpe_de_tokenizer(train_data, save_dir):
    special_tokens = ["PAD", "UNK", "UTT_BOUNDARY"]
    
    tokenizer = Tokenizer(models.BPE())
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
    
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="PAD",
        bos_token="UTT_BOUNDARY",
        eos_token="UTT_BOUNDARY"
    )

    wrapped_tokenizer.pad_token = wrapped_tokenizer.eos_token

    bos_id = wrapped_tokenizer.bos_token_id
    eos_id = wrapped_tokenizer.eos_token_id
    wrapped_tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{wrapped_tokenizer.bos_token} $A {wrapped_tokenizer.eos_token}",
        pair=f"{wrapped_tokenizer.bos_token} $A {wrapped_tokenizer.eos_token} {wrapped_tokenizer.bos_token} $B {wrapped_tokenizer.eos_token}",
        special_tokens=[
            (wrapped_tokenizer.bos_token, bos_id),
            (wrapped_tokenizer.eos_token, eos_id),
            (wrapped_tokenizer.pad_token, wrapped_tokenizer.pad_token_id),
        ],
    )
    
    os.makedirs(save_dir, exist_ok=True)
    wrapped_tokenizer.save_pretrained(save_dir)


# Trainieren & speichern
gen_bpe_de_tokenizer("data/baby_LM/trimmed_babylm_de.txt", "data/tokenizer/bpe_de_tokenizer")
gen_bpe_de_tokenizer("data/baby_LM/trimmed_babylm_en.txt", "data/tokenizer/bpe_en_tokenizer")

# Laden & testen
bpe_tokenizer = PreTrainedTokenizerFast.from_pretrained("data/tokenizer/bpe_de_tokenizer")

# Test
text = "Überraschung! Das ist überraschend"
out = bpe_tokenizer(text, add_special_tokens=True)  # Special Token am Anfang

print("=== Input IDs & Tokens ===")
for idx in out["input_ids"]:
    token = bpe_tokenizer.convert_ids_to_tokens(idx)
    print(f"{idx:5}  {token!r}")

decoded_text = bpe_tokenizer.decode(out["input_ids"], skip_special_tokens=False )
print("\n=== Decoded Text ===")
print(decoded_text)

# Prüfen, welche Special Tokens der Tokenizer kennt
print("\n=== Special Tokens ===")
print("All Special Tokens:", bpe_tokenizer.all_special_tokens)
print("All Special Token IDs:", bpe_tokenizer.all_special_ids)
print("BOS Token:", bpe_tokenizer.bos_token, "ID:", bpe_tokenizer.bos_token_id)
print("EOS Token:", bpe_tokenizer.eos_token, "ID:", bpe_tokenizer.eos_token_id)
print("PAD Token:", bpe_tokenizer.pad_token, "ID:", bpe_tokenizer.pad_token_id)
print("UNK Token:", bpe_tokenizer.unk_token, "ID:", bpe_tokenizer.unk_token_id)