from razdel import tokenize
from corus.sources.rudrec import RuDReCRecord, load_rudrec
import pandas as pd
import numpy as np


class MednerDataset:
    def __init__(
        self,
        rudrec_path: str,
        caws_df_path: str,
        caws_labels_path: str,
        hand_labeled_caws_df_path: str,
        default_entity_type: str = "DI"
    ):  
        self.rudrec = list(load_rudrec(rudrec_path))
        self.caws_df = self.load_caws_df(caws_df_path)
        self.caws_labels = self.load_caws_labels(caws_labels_path)
        self.hand_labeled_caws_df = self.load_hand_labeled_caws_df(hand_labeled_caws_df_path)
        self.default_entity_type = default_entity_type
        self.medical_data = self.collate_data()
    
    def __len__(self):
        return len(self.medical_data)
    
    def __getitem__(self, idx: int) -> dict[str, list[str]]:
        return self.medical_data[idx]
    
    @staticmethod
    def load_caws_df(caws_df_path: str) -> pd.DataFrame:
        return pd.read_csv(caws_df_path)
    
    @staticmethod
    def load_caws_labels(caws_labels_path: str) -> pd.DataFrame:
        return pd.read_json(caws_labels_path).T
   
    @staticmethod
    def load_hand_labeled_caws_df(hand_labeled_caws_df_path: str) -> pd.DataFrame:
        hand_labeled_caws_df = pd.read_json(hand_labeled_caws_df_path)
        hand_labeled_caws_df['id'] = (hand_labeled_caws_df['id'] - np.ones(len(hand_labeled_caws_df['id']))).astype(int)
        hand_labeled_caws_df.drop([0, 239], inplace=True)
        return hand_labeled_caws_df
    
    @staticmethod
    def tokenize_text(text: str):
        raw_tokens = list(tokenize(text))
        words = [token.text for token in raw_tokens]
        
        word_labels = ['O'] * len(raw_tokens)
        char2word = [None] * len(text)
        
        for i, word in enumerate(raw_tokens):
            char2word[word.start:word.stop] = [i] * len(word.text)
            
        return words, word_labels, char2word
            
    
    def extract_labels_rudrec(self, item: RuDReCRecord) -> dict[str, list[str]] | None:
        words, word_labels, char2word = self.tokenize_text(item.text)
        
        for e in item.entities:
            if e.entity_type in ['ADR', 'DI']:
                e_words = sorted({idx for idx in char2word[e.start:e.end] if idx is not None})
                word_labels[e_words[0]] = 'B-' + self.default_entity_type
                for idx in e_words[1:]:
                    word_labels[idx] = 'I-' + self.default_entity_type

        if len(set(word_labels)) != 1:
            return {'tokens': words, 'tags': word_labels}
        else:
            return None
        
    def extract_labels_caws(
        self,
        text: str,
        spans: list[list[int]] | list[dict],
        data_type: str = 'hand_labeled',
    ) -> dict[str, list[str]]:
        words, word_labels, char2word = self.tokenize_text(text)

        for e in spans:
            if data_type == 'hand_labeled':
                e = e['value']
                span_start, span_end = e['start'], e['end']
            else:
                span_start, span_end = e[0], e[1]

            e_words = sorted({idx for idx in char2word[span_start:span_end] if idx is not None})
            word_labels[e_words[0]] = 'B-' + self.default_entity_type
            for idx in e_words[1:]:
                word_labels[idx] = 'I-' + self.default_entity_type

        return {'tokens': words, 'tags': word_labels}

    def process_rudrec(self) -> list[dict[str, list[str]]]:
        processed_data = []
        for item in self.rudrec:
            extracted_data = self.extract_labels_rudrec(item)
            if extracted_data:
                processed_data.append(extracted_data)

        return processed_data
    
    def process_caws(self) -> list[dict[str, list[str]]]:
        processed_data = []
        for i, row in self.caws_df.iloc[:30].iterrows():
            extracted_data = self.extract_labels_caws(row['text'],
                                                self.caws_labels['span'][i],
                                                data_type='caws')
            processed_data.append(extracted_data)

        for _, row in self.hand_labeled_caws_df.iterrows():
            extracted_data = self.extract_labels_caws(row['data']['text'],
                                                row['annotations'][0]['result'],
                                                data_type='hand_labeled')
            processed_data.append(extracted_data)

        return processed_data
    
    def collate_data(self) -> list[dict[str, list[str]]]:
        rudrec_data = self.process_rudrec()
        caws_data = self.process_caws()
        medical_data = rudrec_data + caws_data
        
        return medical_data
    
    def get_medical_data(self, dataframe=True, shuffle=False) -> pd.DataFrame | list[dict[str, list[str]]]:
        if shuffle:
            medical_data = np.random.permutation(self.medical_data).tolist()
        else:
            medical_data = self.medical_data
            
        if dataframe:
            return pd.DataFrame(medical_data)
        else:
            return medical_data
    

    