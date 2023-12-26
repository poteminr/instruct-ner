<div align="center">

![image](images/logo.png "Title")
# Instruct NER

Solution of complex [Named Entity Recognition](https://paperswithcode.com/task/named-entity-recognition-ner) tasks (and subtask [Nested NER](https://paperswithcode.com/task/nested-named-entity-recognition)) based on modern Large Language Models (LLMs). 
</div>

## Table of contents

- [Insturct Dataset](#insturct-dataset)
- [Automatic calculation of metrics](#automatic-calculation-of-metrics)
- [Results](#results)
    - [Restrictions](#restrictions)
- [Models](#models)


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
## Automatic calculation of metrics
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
 ### 1. [Russian Drug Reaction Corpus (RuDReC)](https://github.com/cimm-kzn/RuDReC)

 ### Llama2 7B with LoRA
 ||Drugname|Drugclass|Drugform|DI|ADR|Finding|
|---|---|---|---|---|---|---|
|precision|0.967025|0.995604|0.955377|0.875091|0.998498|0.961349|
|recall|0.919564|0.938536|0.901311|0.797073|0.849298|0.956643|
|f1|0.942697|0.966228|0.927557|0.834262|0.917874|0.958991|

### Mistral 7B with LoRA
|           |   Drugname |   Drugclass |   Drugform |       DI |      ADR |   Finding |
|:----------|-----------:|------------:|-----------:|---------:|---------:|----------:|
| precision |   0.982337 |    0.999309 |   0.989719 | 0.907955 | 0.960243 |  0.981625 |
| recall    |   0.979675 |    0.995871 |   0.987013 | 0.897919 | 0.921734 |  0.960581 |
| f1        |   0.981004 |    0.997587 |   0.988364 | 0.902909 | 0.940594 |  0.970989 |


### rubert-tiny2 29.4M
||Drugname|Drugclass|Drugform|DI|ADR|Finding|
|---|---|---|---|---|---|---|
|precision|0.941215|0.991730|0.989078|0.876774|0.897867|0.977305|
|recall|0.974692|0.992414|0.993146|0.898215|0.921088|0.958275|
|f1|0.957661|0.992072|0.991108|0.887365|0.909329|0.967697|

### 2. [NEREL-BIO](https://github.com/nerel-ds/NEREL-BIO) (Nested Named Entities)
LLM doesn't produce structured output due to the large tagset (40 entities) and nested subentities. 
### 3. [CoNLL-2003](https://paperswithcode.com/dataset/conll-2003)
### Mistral 7B with LoRA

**Base `insturct-ner` target format**
```python
{'PER': ['Nadim Ladki'], 'ORG': [], 'LOC': [], 'MISC': []}
```
|           |      PER |      ORG |      LOC |     MISC |
|:----------|---------:|---------:|---------:|---------:|
| precision | 0.990027 | 0.950637 | 0.976433 | 0.958751 |
| recall    | 0.98536  | 0.9703   | 0.965061 | 0.959301 |
| f1        | 0.987688 | 0.960367 | 0.970714 | 0.959026 |

**Splitted by words target format**

```python
{'PER': ['Nadim', 'Ladki'], 'ORG': [], 'LOC': [], 'MISC': []}
```
|           |      PER |      ORG |      LOC |     MISC |
|:----------|---------:|---------:|---------:|---------:|
| precision | 0.991059 | 0.945115 | 0.972459 | 0.944682 |
| recall    | 0.988945 | 0.972525 | 0.962816 | 0.955922 |
| f1        | 0.990001 | 0.958624 | 0.967613 | 0.950269 |

<!-- **Base encoder format**

*Postprocessing under construction (!!)*
```python
F1 ≈ 0.915 (estimation)
``` -->

### Restrictions
Instruction LLM for NER performs well on flat entities, but performs poorly on datasets with large tagset and nested entites. It's hard to output all entities in a single response due to performance limitations.

Thus, LLM and encoder model produce comparable results on flat-ner datasets with incredibly different training and inference times. 
## Models
* [poteminr/llama2-rudrec](https://huggingface.co/poteminr/llama2-rudrec) adapter model (LoRA)
* [poteminr/llama2-rudrec-merged](https://huggingface.co/poteminr/llama2-rudrec-merged) merged with [base model](https://huggingface.co/meta-llama/Llama-2-7b-hf) 
* [poteminr/mistral-rudrec](https://huggingface.co/poteminr/mistral-rudrec) adapter model (LoRA)

and other models on HF such as T5 Llama:
[poteminr](https://huggingface.co/poteminr)