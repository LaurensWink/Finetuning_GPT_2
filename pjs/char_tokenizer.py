import os
import string
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors
from tokenizers.normalizers import Lowercase, NFD, Sequence, Strip, Replace, NFC
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer
from tokenizers.pre_tokenizers import Split

def gen_char_tokenizer():
    extra_chars = ["ä", "ö", "ü", "ß"]
    whitespace_token = "W"
    special_tokens = ["PAD", "UNK", "UTT_BOUNDARY", whitespace_token]

    vocab_chars = sorted(set(list(string.ascii_lowercase + string.digits + string.punctuation) + extra_chars))

    vocab = {ch: idx for idx, ch in enumerate(special_tokens + vocab_chars)}

    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="UNK"))

    tokenizer.normalizer = Sequence([
        NFC(),
        Lowercase(),
        Replace(" ", whitespace_token),
    ])

    tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")

    tokenizer.enable_padding(pad_id=vocab["PAD"], pad_token="PAD")
    tokenizer.enable_truncation(max_length=512)

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="UNK",
        pad_token="PAD",
        bos_token="UTT_BOUNDARY",
        eos_token="UTT_BOUNDARY",
    )

    os.makedirs("data/tokenizer/char_tokenizer", exist_ok=True)
    wrapped_tokenizer.save_pretrained("data/tokenizer/char_tokenizer")

gen_char_tokenizer()

my_char_tokenizer = PreTrainedTokenizerFast.from_pretrained("data/tokenizer/char_tokenizer")

text = "Überraschung! Das ist überraschend"

my_output = my_char_tokenizer(text, add_special_tokens=True)

# IDs ausgeben
print("=== Eigener Char-Tokenizer ===")
print("Input IDs:", my_output["input_ids"])
print("Tokens:   ", my_char_tokenizer.convert_ids_to_tokens(my_output["input_ids"]))
