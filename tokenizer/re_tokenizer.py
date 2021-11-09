from dataclasses import dataclass
from . import KoBertTokenizer
from transformers import ElectraTokenizer

@dataclass
class ReTokenizerFactories:
    BERT_TOKENIZER: KoBertTokenizer

class ReTokenizer:
    def __init__(self, tokenizer_name=None):
        self.tokenizer_name = tokenizer_name

        if tokenizer_name == "BERT" or tokenizer_name == "DISTILBERT":
            self.tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
        elif tokenizer_name == "ELECTRA":
            self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def get_pad_token(self):
        return self.tokenizer.pad_token

    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id

    def __call__(self, data, method):
        rel_label = data[-1]

        if method == "typed_entity_marker":
            token_list = self.toeknize_temp(data)
        else:
            raise NotImplementedError()

        return token_list, rel_label

    def tokenize_emr(self, data):
        # [E1]Bill[/E1] was born in [E2] Seattle [/E2].
        # TODO
        sentence, subj_start, subj_end, subj_label, obj_start, obj_end, obj_label, rel_label = data

    def toeknize_temp(self, data):
        # @ * person * Bill @ was born in # ^ city ^ Seattle #.
        # TODO
        sentence, subj_start, subj_end, subj_label, obj_start, obj_end, obj_label, _ = data

        subj_start = int(subj_start)
        subj_end = int(subj_end)

        obj_start = int(obj_start)
        obj_end = int(obj_end)

        if subj_start < obj_start:
            first_sentence = sentence[:obj_start]
            obj_word = sentence[obj_start: obj_end]
            last_sentence = sentence[obj_end: ]

            sentence = first_sentence + f"# ^ {obj_label} ^ {obj_word} # " + last_sentence

            first_sentence = sentence[:subj_start]
            subj_word = sentence[subj_start: subj_end]
            last_sentence = sentence[subj_end:]

            sentence = first_sentence + f"@ * {subj_label} * {subj_word} @ " + last_sentence
        else:
            first_sentence = sentence[:subj_start]
            subj_word = sentence[subj_start: subj_end]
            last_sentence = sentence[subj_end:]

            sentence = first_sentence + f"@ * {subj_label} * {subj_word} @ " + last_sentence

            first_sentence = sentence[:obj_start]
            obj_word = sentence[obj_start: obj_end]
            last_sentence = sentence[obj_end:]

            sentence = first_sentence + f"# ^ {obj_label} ^ {obj_word} # " + last_sentence

        return self.tokenize(sentence=sentence)