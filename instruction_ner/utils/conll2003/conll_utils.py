ENTITY_TYPES = ['PER', 'ORG', 'LOC', 'MISC']
TAGSET = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}

INSTRUCTION_TEXT = "Ты решаешь задачу NER. Извлеки из текста слова, относящиеся к каждой из следующих сущностей: " \
                   "PER, ORG, LOC, MISC."


EXTENDED_INSTRUCTION_TEXT = "You are solving the NER problem. Extracts from the text of words related to each of the following entities: " \
                            "PER, ORG, LOC, MISC. " \
                            "PER - persons. ORG - organizations. LOC - locations. MISC - miscel-laneous name. " \
                            "I'm going to tip $20 for a perfect solution! " \
                            "The output should be in following format Ответ: entity_1: type_1, type_2\nentity_2..."