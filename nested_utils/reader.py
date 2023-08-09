from pybrat.parser import BratParser, Example, Entity
import numpy as np
from collections import defaultdict
from razdel import sentenize
from razdel.substring import Substring


def parse_examples(data_path: str) -> list[Example]:
    parser = BratParser(ignore_types=['R'], error="ignore")
    return parser.parse(data_path)


def split_example(example: Example, n_splits: int = 4) -> tuple[list[list[Substring]], dict[int, list[Entity]]]:
    sentences = list(sentenize(example.text))
    entities = sorted(example.entities, key=lambda x: x.spans[0].start)
    
    splited_sentences = np.array_split(np.array(sentences), n_splits)
    boundings = [(a[0].start, a[-1].stop) for a in splited_sentences]
    
    splited_entities = defaultdict(list)
    
    for entity in entities:
        for bound_index, bound in enumerate(boundings):
            if entity.spans[0].start >= bound[0] and entity.spans[0].end <= bound[1]:
                splited_entities[bound_index].append(entity)
                break
                
    return splited_sentences, splited_entities