<div align="center">

![image](images/logo.png "Title")
# Instruct NER

Solution of complex [Named Entity Recognition](https://paperswithcode.com/task/named-entity-recognition-ner) tasks (and subtask [Nested NER](https://paperswithcode.com/task/nested-named-entity-recognition)) based on modern Large Language Models (LLMs). 
</div>

## Table of contents

- [Insturct Dataset](#insturct-dataset)
    - [Implemented datasets](#implemented-datasets)
    - [Training](#train-your-llm-on-instructions)
- [Automatic calculation of metrics](#automatic-calculation-of-metrics)
    - [Inference](#infer-your-llm-on-instructions-to-generate-predictionjson)
- [Results](#results)
    - [Metrics](#tables-with-metrics-for-implemented-datasets-link)
    - [Error analysis](#error-analysis-link)
    - [Restrictions](#restrictions)
- [Models](#models)
    - [Implemented models](#implemented-models)
    - [HuggingFace](#huggingface)


## Insturct Dataset
You should form python dictionaries for every text and labels. Let's look at an simplified example from [Russian Drug Reaction Corpus (RuDReC)](https://github.com/cimm-kzn/RuDReC).

* Input text: `Это старый-добрый Римантадин, только в сиропе.`
* Labels: `Римантадин - Drugname, сиропе - Drugform` 

### 1. Create `Instruction` - task description for LLM 
Russian:
>Ты решаешь задачу NER. Извлеки из текста слова, относящиеся к каждой из следующих сущностей: Drugname, Drugclass, DI, ADR, Finding.

English:
>You are solving the NER problem. Extract from the text words related to each of the following entities: Drugname, Drugclass, DI, ADR, Finding.

### 2. Build `dictionary with labels`. 
You can use one of two supported version.

#### With all entity types (hard to compute with large tagset)
```python
raw_entities = {
    'Drugname': ['Римантадин'],
    'Drugclass': [],
    'Drugform': ['сиропе'],
    'DI': [],
    'ADR': [],
    'Finding': []
}
```
#### Only with mentioned entities (better for large tagset)
```python
short_form_output=True (available with Nerel-BIO and MultiCoNER)
```

```python
raw_entities = {
    'Drugname': ['Римантадин'],
    'Drugform': ['сиропе']
}
```

### 3. Create `MODEL_INPUT_TEMPLATE`.

```python
MODEL_INPUT_TEMPLATE = {
'prompts_input': "### Задание: {instruction}\n### Вход: {inp}\n### Ответ: ",
'output_separator': "Ответ: "
}
```
Or english version
```python
MODEL_INPUT_TEMPLATE = {
'prompts_input': "### Task: {instruction}\n### Input: {inp}\n### Answer: ",
'output_separator': "Answer: "
}
```

### Automatically generate `Instruction`
 `instruction_ner/utils/instruct_dataset.py`
```python
class Instruction(TypedDict):
    instruction: str
    input: str
    output: str
    source: str   
    raw_entities: dict[str, list[str]]
    id: str
```
#### Example
```python
{'instruction': 'Ты решаешь задачу NER. Извлеки из текста слова, относящиеся к каждой из следующих сущностей: Drugname, Drugclass, DI, ADR, Finding.',
 'input': 'Это старый-добрый Римантадин, только в сиропе.\n',
 'output': 'Drugname: Римантадин\nDrugclass: \nDrugform: сиропе\nDI: \nADR: \nFinding: \n',
 'source': '### Задание: Ты решаешь задачу NER. Извлеки из текста слова, относящиеся к каждой из следующих сущностей: Drugname, Drugclass, DI, ADR, Finding.\n### Вход: Это старый-добрый Римантадин, только в сиропе.\n### Ответ: ',
 'raw_entities': {'Drugname': ['Римантадин'],
  'Drugclass': [],
  'Drugform': ['сиропе'],
  'DI': [],
  'ADR': [],
  'Finding': []},
 'id': '1_2555494.tsv'}
 ```
### Implemented datasets
`instruction_ner/utils/`
1. [Russian Drug Reaction Corpus (RuDReC)](https://github.com/cimm-kzn/RuDReC)
2. [NEREL-BIO](https://github.com/nerel-ds/NEREL-BIO) (Nested Named Entities)
3. [CoNLL-2003](https://paperswithcode.com/dataset/conll-2003)
4. [MultiCoNER II (2023)](https://multiconer.github.io/dataset) ([HF](https://huggingface.co/datasets/MultiCoNER/multiconer_v2/viewer/English%20(EN)), *fine and coarse level mapping of the tags*)

### Train your LLM on `instructions`  
```python
python medner/instruction_ner/train_instruct.py \
        --config_file medner/instruction_ner/configs/mistral_7b.json \
        --model_type mistral \
        --dataset_name conll2003 \
        --max_instances -1 \
        --push_to_hub True \
        --hf_name_postfix _extended_instruction
```

## Automatic calculation of metrics
### Infer your LLM on `instructions` to generate `prediction.json`
```python
python medner/instruction_ner/inference_instruct.py \
        --batch_size 16 \
        --dataset_name conll2003 \
        --model_type mistral \
        --model_name poteminr/mistral-conll2003_extended_instruction \
        --max_instances -1
```
`instruction_ner/metric.py`

You can use the implemented functions with the output of inference_instruct calculate metrics. 
```python
import pandas as pd
from utils.rudrec.rudrec_utis import ENTITY_TYPES
from metric import calculate_metrics_from_dataframe

prediction = pd.read_json('prediction.json')
prediction.head(3)
```
|    | id            | extracted                                                                                                 | target                                                                                                    |
|---:|:--------------|:----------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------|
|  0 | 8_1443820.tsv | {'Drugname': [], 'Drugclass': [], 'Drugform': ['таблетки'], 'DI': [], 'ADR': [], 'Finding': []}           | {'Drugname': [], 'Drugclass': [], 'Drugform': ['таблетки'], 'DI': [], 'ADR': [], 'Finding': []}           |
|  1 | 1_2555494.tsv | {'Drugname': ['Римантадин'], 'Drugclass': [], 'Drugform': ['сиропе'], 'DI': [], 'ADR': [], 'Finding': []} | {'Drugname': ['Римантадин'], 'Drugclass': [], 'Drugform': ['сиропе'], 'DI': [], 'ADR': [], 'Finding': []} |
|  2 | 1_618967.tsv  | {'Drugname': [], 'Drugclass': [], 'Drugform': [], 'DI': [], 'ADR': [], 'Finding': []}                     | {'Drugname': [], 'Drugclass': [], 'Drugform': [], 'DI': [], 'ADR': [], 'Finding': []}                     |

```python
from metric import calculate_metrics_from_dataframe
metrics = calculate_metrics_from_dataframe(prediction, ENTITY_TYPES)
```
```python
{'Drugname': {'precision': 0.9670250896057347,
  'recall': 0.9195637355146558,
  'f1': 0.9426974143955277}, ...}
```
## Results
### [Tables with metrics for implemented datasets (link)](METRICS.md)
### Error analysis [(link)](instruction_ner/error_analysis/README.md)
You can explore 5 types of model errors:
1. **Mistaken recognition** - one type of entity is recognized as another
2. **Entity is not recognized**
3. **Misspelling** - origin text doesn't contain the predicted entity
4. **Overpredictiton**
5. **[Conflicting predictions](https://github.com/universal-ner/universal-ner/issues/19)**

*Confusion matrix for mistaken recognitions  is available.*
### Restrictions
Instruction LLM for NER performs well on flat entities, but performs poorly on datasets with large tagset and nested entites. 

Thus, LLM and encoder model produce comparable results on flat-ner datasets with incredibly different training and inference times. 
## Models
### Implemented models
1. Llama & Llama2
2. Mistral
3. T5
4. RWKV
### HuggingFace
* [poteminr/llama2-rudrec](https://huggingface.co/poteminr/llama2-rudrec) adapter model (LoRA)
* [poteminr/llama2-rudrec-merged](https://huggingface.co/poteminr/llama2-rudrec-merged) merged with [base model](https://huggingface.co/meta-llama/Llama-2-7b-hf) 
* [poteminr/mistral-rudrec](https://huggingface.co/poteminr/mistral-rudrec) adapter model (LoRA)

and other models on HF such as T5, Llama, Mistral:
[poteminr](https://huggingface.co/poteminr)