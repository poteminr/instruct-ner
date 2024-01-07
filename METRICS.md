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
|           |      PER |      ORG |      LOC |     MISC |   overall |
|:----------|---------:|---------:|---------:|---------:|----------:|
| precision | 0.990027 | 0.950637 | 0.976433 | 0.958751 |  0.969374 |
| recall    | 0.98536  | 0.9703   | 0.965061 | 0.959301 |  0.970467 |
| f1        | 0.987688 | 0.960367 | 0.970714 | 0.959026 |  0.96992  |

**Splitted by words target format**
```python
split_entities=True (instruction_ner/metric.py)
```

```python
{'PER': ['Nadim', 'Ladki'], 'ORG': [], 'LOC': [], 'MISC': []}
```
|           |      PER |      ORG |      LOC |     MISC |   overall |
|:----------|---------:|---------:|---------:|---------:|----------:|
| precision | 0.991059 | 0.945115 | 0.972459 | 0.944682 |  0.964659 |
| recall    | 0.988945 | 0.972525 | 0.962816 | 0.955922 |  0.971615 |
| f1        | 0.990001 | 0.958624 | 0.967613 | 0.950269 |  0.968125 |

<!-- **Base encoder format**

*Postprocessing under construction (!!)*
```python
F1 ≈ 0.915 (estimation)
``` -->
### 4. [MultiCoNER II (2023)](https://huggingface.co/datasets/MultiCoNER/multiconer_v2/viewer/English%20(EN))
* English (test)
* Shuffled with seed 42
* First 10k samples

[The fine to coarse level mapping of the tags (link)](https://multiconer.github.io/dataset)
### Mistral 7B with LoRA
#### Coarse tagset
|           |      LOC |       CW |      GRP |      PER |     PROD |      MED |   overall |
|:----------|---------:|---------:|---------:|---------:|---------:|---------:|----------:|
| precision | 0.915037 | 0.937586 | 0.95428  | 0.961388 | 0.964142 | 0.966908 |  0.949754 |
| recall    | 0.939475 | 0.942131 | 0.938311 | 0.965418 | 0.950584 | 0.966716 |  0.9507   |
| f1        | 0.927095 | 0.939853 | 0.946228 | 0.963399 | 0.957315 | 0.966812 |  0.950226 |
#### Fine tagset
|           |   overall |
|:----------|----------:|
| precision |  0.98425  |
| recall    |  0.984075 |
| f1        |  0.984162 |