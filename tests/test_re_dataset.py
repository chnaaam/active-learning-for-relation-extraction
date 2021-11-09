import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.re_dataset import ReDataset
from tokenizer.re_tokenizer import ReTokenizer

if __name__ == "__main__":
    tokenizer = ReTokenizer(tokenizer_name="BERT")
    
    re_dataset = ReDataset(
        vocab_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vocab"),
        tokenizer=tokenizer,
        model_type="BERT",
        dataset_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        dataset_fn="train.json",
        cache_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"),
        max_len=100)

    # print(re_dataset.token_list[-1])
    # print(re_dataset.label_list[-1])
    #
    # print(tokenizer.tokenizer.convert_tokens_to_string(re_dataset.token_list[-1]))