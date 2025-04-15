from pjs.dataset import Data
from transformers import AutoTokenizer

data = Data("data/LMentry")
data.split(0.8)
tokenizer = AutoTokenizer.from_pretrained('phonemetransformers/babble-tokenizers', subfolder='BABYLM-TOKENIZER-CHAR-TXT')
print(data.get_tokenised_split(tokenizer))