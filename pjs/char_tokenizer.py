import os
import string
from tokenizers import Tokenizer, models
from tokenizers.normalizers import Lowercase, Sequence, Replace, NFC
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing

def gen_char_tokenizer():
    tokenizer = Tokenizer(models.BPE())
    whitespace_token = "W"
    tokenizer.normalizer = Sequence([
        NFC(),
        Lowercase(),
        Replace(" ", whitespace_token),
    ])
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="PAD",
        bos_token="UTT_BOUNDARY",
        eos_token="UTT_BOUNDARY",
    )

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

    os.makedirs("data/tokenizer/char_tokenizer", exist_ok=True)
    wrapped_tokenizer.save_pretrained("data/tokenizer/char_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("data/tokenizer/char_tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer._tokenizer.get_vocab_size(with_added_tokens=True)
    ascii_list = []
    ascii_string = string.printable
    for char in ascii_string:
        ascii_list.append(char)
    tokenizer.add_tokens(ascii_list)
    tokenizer.add_tokens(['ö','Ö','ä','Ä','ü','Ü','ß','ẞ'])

    tokenizer.save_pretrained("data/tokenizer/char_tokenizer")

gen_char_tokenizer()
char_tokenizer = AutoTokenizer.from_pretrained("data/tokenizer/char_tokenizer")

# Test
text = "Überraschung! Das ist überraschend"
out = char_tokenizer(text, add_special_tokens=True)  # Special Token am Anfang

print("=== Input IDs & Tokens ===")
for idx in out["input_ids"]:
    token = char_tokenizer.convert_ids_to_tokens(idx)
    print(f"{idx:5}  {token!r}")

decoded_text = char_tokenizer.decode(out["input_ids"], skip_special_tokens=False )
print("\n=== Decoded Text ===")
print(decoded_text)

# Prüfen, welche Special Tokens der Tokenizer kennt
print("\n=== Special Tokens ===")
print("All Special Tokens:", char_tokenizer.all_special_tokens)
print("All Special Token IDs:", char_tokenizer.all_special_ids)
print("BOS Token:", char_tokenizer.bos_token, "ID:", char_tokenizer.bos_token_id)
print("EOS Token:", char_tokenizer.eos_token, "ID:", char_tokenizer.eos_token_id)
print("PAD Token:", char_tokenizer.pad_token, "ID:", char_tokenizer.pad_token_id)
print("UNK Token:", char_tokenizer.unk_token, "ID:", char_tokenizer.unk_token_id)