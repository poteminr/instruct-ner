### 1. [Russian Drug Reaction Corpus (RuDReC)](https://github.com/cimm-kzn/RuDReC)
 ### Llama2 7B with LoRA
|           |   Drugname |   Drugclass |   Drugform |       DI |        ADR |   Finding |   overall |
|:----------|-----------:|------------:|-----------:|---------:|-----------:|----------:|----------:|
| precision |   0.793722 |   0.5       |   0.636905 | 0.408304 | 0.333333   |  0.166667 |  0.551905 |
| recall    |   0.6      |   0.0631579 |   0.428    | 0.27896  | 0.00421941 |  0.150685 |  0.305899 |
| f1        |   0.683398 |   0.11215   |   0.511962 | 0.331461 | 0.00833333 |  0.158273 |  0.393627 |


### Mistral 7B with LoRA
|           |   Drugname |   Drugclass |   Drugform |       DI |      ADR |   Finding |   overall |
|:----------|-----------:|------------:|-----------:|---------:|---------:|----------:|----------:|
| precision |   0.910653 |    0.988889 |   0.939024 | 0.655172 | 0.662857 |  0.380952 |  0.7864   |
| recall    |   0.898305 |    0.936842 |   0.924    | 0.628842 | 0.489451 |  0.219178 |  0.71595  |
| f1        |   0.904437 |    0.962162 |   0.931452 | 0.641737 | 0.563107 |  0.278261 |  0.749523 |



### rubert-tiny2 29.4M (encoder)
|           |   Drugname |   Drugclass |   Drugform |       DI |      ADR |   Finding |   overall |
|:----------|-----------:|------------:|-----------:|---------:|---------:|----------:|----------:|
| precision |   0.774481 |    0.884211 |   0.926923 | 0.533589 | 0.368771 |  0.529412 |  0.642717 |
| recall    |   0.884746 |    0.884211 |   0.964    | 0.65721  | 0.468354 |  0.123288 |  0.716679 |
| f1        |   0.825949 |    0.884211 |   0.945098 | 0.588983 | 0.412639 |  0.2      |  0.677686 |

### 2. [NEREL-BIO](https://github.com/nerel-ds/NEREL-BIO) (Nested Named Entities)
LLM doesn't produce structured output due to the large tagset (40 entities) and nested subentities. 
### 3. [CoNLL-2003](https://paperswithcode.com/dataset/conll-2003)
### Mistral 7B with LoRA

**Default `insturct-ner` target format (exact match)**
```python
{'PER': ['Nadim Ladki'], 'ORG': [], 'LOC': [], 'MISC': []}
```
|           |      PER |      ORG |      LOC |     MISC |   overall |
|:----------|---------:|---------:|---------:|---------:|----------:|
| precision | 0.974922 | 0.889148 | 0.945    | 0.791908 |  0.917515 |
| recall    | 0.963445 | 0.93154  | 0.919708 | 0.794203 |  0.920308 |
| f1        | 0.969149 | 0.909851 | 0.932182 | 0.793054 |  0.918909 |


**Splitted by words target format (partial match)**
```python
split_entities=True (instruction_ner/metric.py)
```

```python
{'PER': ['Nadim', 'Ladki'], 'ORG': [], 'LOC': [], 'MISC': []}
```
|           |      PER |      ORG |      LOC |     MISC |   overall |
|:----------|---------:|---------:|---------:|---------:|----------:|
| precision | 0.983144 | 0.899768 | 0.94086  | 0.780488 |  0.923361 |
| recall    | 0.979197 | 0.948592 | 0.921053 | 0.818687 |  0.937922 |
| f1        | 0.981167 | 0.923535 | 0.930851 | 0.799131 |  0.930585 |


<!-- **Base encoder format**

*Postprocessing under construction (!!)*
```python
F1 ≈ 0.915 (estimation)
``` -->
### 4. [MultiCoNER II (2023)](https://huggingface.co/datasets/MultiCoNER/multiconer_v2/viewer/English%20(EN))
* English (test)
* Shuffled with seed 42
* First 10k test samples (due to inferece time)

[The fine to coarse level mapping of the tags (link)](https://multiconer.github.io/dataset)
### Mistral 7B with LoRA
#### Coarse tagset
|           |      LOC |       CW |      GRP |      PER |     PROD |      MED |   overall |
|:----------|---------:|---------:|---------:|---------:|---------:|---------:|----------:|
| precision | 0.692308 | 0.748811 | 0.792315 | 0.921221 | 0.647929 | 0.622877 |  0.793998 |
| recall    | 0.76431  | 0.763636 | 0.735456 | 0.929134 | 0.568339 | 0.621469 |  0.797251 |
| f1        | 0.726529 | 0.756151 | 0.762828 | 0.92516  | 0.60553  | 0.622172 |  0.795621 |

#### Fine tagset
|           |   overall |
|:----------|----------:|
| precision |  0.624569 |
| recall    |  0.621927 |
| f1        |  0.623245 |