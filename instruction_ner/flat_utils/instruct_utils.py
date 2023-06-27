SEP_SYMBOL = ' | '
ENTITY_TYPES = ['Drugname', 'Drugclass', 'Drugform', 'DI', 'ADR', 'Finding']
ENTITY_DEFENITIONS = [
    'упоминания торговой марки лекарства или ингредиенты/активные соединения продукта',
    'упоминания классов припаратов, таких как противовоспалительные или сердечно-сосудистые',
    'упоминания о способах введения, таких как таблетка или жидкости, которые описывают физическую форму лекарства',
    'любое указание/симптом, указывающий на причину назначения препарата',
    'упоминания о неблагоприятных медицинских событиях, которые происходят как следствие приема лекарств, '
    'и не связаны с излечиваемыми симптомами (побочные эффекты)',
    'любые побочные эффекты или симптомы, которые не были непосредственно испытаны пациентом'
    ]

MODEL_INPUT_TEMPLATE = {
            'prompts_input': "### Задание: {instruction}\n### Вход: {inp}\n### Ответ: ",
            'output_separator':  "Ответ: "        
        }

GENERAL_INSTRUCTION = "Ты решаешь задачу NER. Извлеки из текста слова, относящиеся к каждой из следующих сущностей: Drugname, Drugclass, DI, ADR, Finding."


def create_output_from_entities(entities: list, out_type: int = 1) -> str:
    if out_type == 1:
        return "{}".format(entities)[1:-1]
    elif out_type == 2:
        return SEP_SYMBOL.join(entities)


def entity_type_to_instruction(entity_type: str) -> str:
    base_phrase = 'Ты решаешь задачу NER. Извлеки из текста '
    return base_phrase + dict(zip(ENTITY_TYPES, ENTITY_DEFENITIONS))[entity_type]
    