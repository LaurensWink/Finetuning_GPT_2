from tokenizers import Tokenizer, models
from tokenizers.normalizers import Lowercase, Sequence, Replace, NFC, Strip
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, GPT2TokenizerFast, LlamaTokenizerFast, PreTrainedTokenizerFast
import os, string
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
        Strip(),
        Replace(" ", whitespace_token),
    ])

    tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")

    pad_id = tokenizer.token_to_id("PAD")
    tokenizer.enable_padding(pad_id=pad_id, pad_token="PAD")
    tokenizer.enable_truncation(max_length=256)


    wrapped_tokenizer = LlamaTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="UNK",
        pad_token="PAD",
        bos_token="UTT_BOUNDARY",
        eos_token="UTT_BOUNDARY"
    )

    os.makedirs("data/tokenizer/char_tokenizer", exist_ok=True)
    wrapped_tokenizer.save_pretrained("data/tokenizer/char_tokenizer")

gen_char_tokenizer()