MODEL_INPUT_TEMPLATE = {
            'prompts_input': "### Задание: {instruction}\n### Вход: {inp}\n### Ответ: ",
            'output_separator':  "Ответ: "        
        }


def create_output_from_entities(entities: dict[str, list], out_type: int = 1) -> str:
    if out_type == 1:
        return ", ".join(entities)
    elif out_type == 2:
        out = ""
        for entity_type in entities.keys():
            out += (f"{entity_type}: " + ", ".join(entities[entity_type]) + "\n")
        return out
