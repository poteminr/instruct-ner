<div align="center">

![image](images/medner_logo_v1.png "Title")
# Instruct NER

Solution of complex [Named Entity Recognition](https://paperswithcode.com/task/named-entity-recognition-ner) tasks (and subtask [Nested NER](https://paperswithcode.com/task/nested-named-entity-recognition)) based on modern Large Language Models (LLMs). 
</div>

## Table of contents

- [Insturct Dataset](#insturct-dataset)
- [Results](#results)
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
#### Only with mentioned entities (better for large tagset).

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

 ## Results
 ### 1. [Russian Drug Reaction Corpus (RuDReC)](https://github.com/cimm-kzn/RuDReC)

 ### Llama2 7B with LoRA
 ||Drugname|Drugclass|Drugform|DI|ADR|Finding|
|---|---|---|---|---|---|---|
|precision|0.967025|0.995604|0.955377|0.875091|0.998498|0.961349|
|recall|0.919564|0.938536|0.901311|0.797073|0.849298|0.956643|
|f1|0.942697|0.966228|0.927557|0.834262|0.917874|0.958991|

### rubert-tiny2 29.4M
||Drugname|Drugclass|Drugform|DI|ADR|Finding|
|---|---|---|---|---|---|---|
|precision|0.941215|0.991730|0.989078|0.876774|0.897867|0.977305|
|recall|0.974692|0.992414|0.993146|0.898215|0.921088|0.958275|
|f1|0.957661|0.992072|0.991108|0.887365|0.909329|0.967697|

### 2. [NEREL-BIO](https://github.com/nerel-ds/NEREL-BIO) (Nested Named Entities)
Soon

## Models
* [poteminr/llama2-rudrec](https://huggingface.co/poteminr/llama2-rudrec) adapter model (LoRA)
* [poteminr/llama2-rudrec-merged](https://huggingface.co/poteminr/llama2-rudrec-merged) merged with [base model](https://huggingface.co/meta-llama/Llama-2-7b-hf) 


and other models on HF such as T5 Llama:
[poteminr](https://huggingface.co/poteminr)