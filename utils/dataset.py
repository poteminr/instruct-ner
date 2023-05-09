import torch
from transformers import AutoTokenizer
from utils.tagset import get_tagset
from pybrat.parser import BratParser
from utils.reader_utils import text_preprocess_function, parse_example


class NerelBioDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dir_dame: str = "data/nerel-bio-v1.0/train/",
                 max_instances: int = -1,
                 encoder_model: str = 'cointegrated/rubert-tiny2',
                 viterbi_algorithm: bool = True,
                 label_pad_token_id: int = -100
                 ):

        self.dir_name = dir_dame
        self.max_instances = max_instances
        self.label_to_id = get_tagset()
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        self.encoder_model = encoder_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_model)

        self.pad_token = self.tokenizer.special_tokens_map['pad_token']
        self.pad_token_id = self.tokenizer.get_vocab()[self.pad_token]
        self.sep_token = self.tokenizer.special_tokens_map['sep_token']

        if viterbi_algorithm:
            self.label_pad_token_id = self.pad_token_id
        else:
            self.label_pad_token_id = label_pad_token_id

        self.instances = []
        self.texts = []
        self.read_data()

    def get_target_size(self):
        return len(set(self.label_to_id.values()))

    def get_target_vocab(self):
        return self.label_to_id

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def read_data(self):
        parser = BratParser(ignore_types=['R'], error="ignore")
        examples = parser.parse(
            self.dir_name,
            text_preprocess_function=text_preprocess_function,
            ann_preprocess_function=text_preprocess_function
        )

        for example in examples:
            self.texts.append(example.text)
            labels, tokenized_inputs = parse_example(example, self.tokenizer)
            input_ids = torch.tensor(tokenized_inputs['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(tokenized_inputs['attention_mask'], dtype=torch.bool)
            self.instances.append((input_ids, attention_mask, labels))

    # def pad_instances(self, input_ids, labels, attention_masks):
    #     max_length_in_batch = max([len(token) for token in input_ids])
    #     input_ids_tensor = torch.empty(size=(len(input_ids), max_length_in_batch), dtype=torch.long).fill_(
    #         self.pad_token_id)
    #     labels_tensor = torch.empty(size=(len(input_ids), max_length_in_batch), dtype=torch.long).fill_(
    #         self.label_pad_token_id)
    #     attention_masks_tensor = torch.zeros(size=(len(input_ids), max_length_in_batch), dtype=torch.bool)
    #
    #     for i in range(len(input_ids)):
    #         tokens_ = input_ids[i]
    #         seq_len = len(tokens_)
    #
    #         input_ids_tensor[i, :seq_len] = tokens_
    #         labels_tensor[i, :seq_len] = labels[i]
    #         attention_masks_tensor[i, :seq_len] = attention_masks[i]
    #
    #     return input_ids_tensor, labels_tensor, attention_masks_tensor
    #
    # def data_collator(self, batch):
    #     batch_ = list(zip(*batch))
    #     input_ids, labels, attention_masks = batch_[0], batch_[1], batch_[2]
    #     return self.pad_instances(input_ids, labels, attention_masks)