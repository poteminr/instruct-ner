import torch
from pybrat.parser import BratParser
from transformers import AutoTokenizer
from nested_utils.reader_utils import text_preprocess_function, parse_example


class NerelBioDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dir_dame: str = "data/nerel-bio-v1.0/train/",
                 encoder_model: str = 'cointegrated/rubert-tiny2',
                 max_instances: int = -1,
                 ):

        self.dir_name = dir_dame
        self.encoder_model = encoder_model
        self.max_instances = max_instances

        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_model)

        self.pad_token = self.tokenizer.special_tokens_map['pad_token']
        self.pad_token_id = self.tokenizer.get_vocab()[self.pad_token]
        self.sep_token = self.tokenizer.special_tokens_map['sep_token']

        self.instances = []
        self.texts = []
        self.read_data()

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
        if self.max_instances != -1:
            examples = examples[:self.max_instances]

        for example in examples:
            self.texts.append(example.text)
            labels, tokenized_inputs = parse_example(example, self.tokenizer)
            input_ids = torch.tensor(tokenized_inputs['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(tokenized_inputs['attention_mask'], dtype=torch.bool)
            self.instances.append((input_ids, labels, attention_mask))

    @staticmethod
    def pad_labels(labels: dict, max_length_in_batch: int, seq_len: int) -> dict:
        padding = torch.zeros((max_length_in_batch-seq_len, 5))
        padding[:, 3] = 1
        for key in labels.keys():
            labels[key] = torch.cat((labels[key], padding))

        return labels

    def pad_instances(self, input_ids, labels, attention_masks):
        labels = list(labels)
        max_length_in_batch = max([len(token) for token in input_ids])
        input_ids_tensor = torch.empty(size=(len(input_ids), max_length_in_batch), dtype=torch.long).fill_(
            self.pad_token_id)
        attention_masks_tensor = torch.zeros(size=(len(input_ids), max_length_in_batch), dtype=torch.bool)

        for i in range(len(input_ids)):
            tokens_ = input_ids[i]
            seq_len = len(tokens_)

            input_ids_tensor[i, :seq_len] = tokens_
            labels[i] = self.pad_labels(labels[i], max_length_in_batch, seq_len)
            attention_masks_tensor[i, :seq_len] = attention_masks[i]

        return input_ids_tensor, labels, attention_masks_tensor

    def data_collator(self, batch):
        batch_ = list(zip(*batch))
        input_ids, labels, attention_masks = batch_[0], batch_[1], batch_[2]
        return self.pad_instances(input_ids, labels, attention_masks)
