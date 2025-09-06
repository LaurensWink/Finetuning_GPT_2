from tokenizers import Tokenizer, models
from tokenizers.normalizers import Lowercase, Sequence, Replace, NFC, Strip
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, GPT2TokenizerFast, LlamaTokenizerFast, PreTrainedTokenizerFast
import os, string
from tokenizers.pre_tokenizers import Split

def gen_char_tokenizer():
    tokenizer = Tokenizer(models.BPE())
    whitespace_token = "W"
    tokenizer.normalizer = Sequence([
        NFC(),
        Lowercase(),
        Replace(" ", whitespace_token),
    ])
    wrapped_tokenizer = LlamaTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="PAD",
        bos_token="UTT_BOUNDARY",
        eos_token="UTT_BOUNDARY",
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
# Erzeugen
wordlevel_tokenizer = AutoTokenizer.from_pretrained("data/tokenizer/char_tokenizer")

# Test
text = "Überraschung! Das ist überraschend"
out = wordlevel_tokenizer(text, add_special_tokens=True)

print("=== Input IDs & Tokens ===")
for idx in out["input_ids"]:
    token = wordlevel_tokenizer.convert_ids_to_tokens(idx)
    print(f"{idx:5}  {token!r}")

decoded_text = wordlevel_tokenizer.decode(out["input_ids"], skip_special_tokens=False)
print("\n=== Decoded Text ===")
print(decoded_text)

print("\n=== Special Tokens ===")
print("All Special Tokens:", wordlevel_tokenizer.all_special_tokens)
print("All Special Token IDs:", wordlevel_tokenizer.all_special_ids)
print("BOS Token:", wordlevel_tokenizer.bos_token, "ID:", wordlevel_tokenizer.bos_token_id)
print("EOS Token:", wordlevel_tokenizer.eos_token, "ID:", wordlevel_tokenizer.eos_token_id)
print("PAD Token:", wordlevel_tokenizer.pad_token, "ID:", wordlevel_tokenizer.pad_token_id)
print("UNK Token:", wordlevel_tokenizer.unk_token, "ID:", wordlevel_tokenizer.unk_token_id)
