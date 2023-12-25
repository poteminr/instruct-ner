ENTITY_TYPES = ['Facility', 'OtherLOC', 'HumanSettlement', 'Station',
                'VisualWork', 'MusicalWork', 'WrittenWork', 'ArtWork', 'Software',
                'OtherCW', 'MusicalGRP', 'PublicCORP', 'PrivateCORP', 'OtherCORP',
                'AerospaceManufacturer', 'SportsGRP', 'CarManufacturer', 'TechCORP',
                'ORG', 'Scientist', 'Artist', 'Athlete', 'Politician', 'Cleric',
                'SportsManager', 'OtherPER', 'Clothing', 'Vehicle', 'Food',
                'Drink', 'OtherPROD', 'Medication/Vaccine', 'MedicalProcedure',
                'AnatomicalStructure', 'Symptom', 'Disease'
                ]

COARSE_ENTITY_TYPES_MAPPING = {'LOC': ['Facility', 'OtherLOC', 'HumanSettlement', 'Station'],
                               'CW': ['VisualWork', 'MusicalWork', 'WrittenWork', 'ArtWork', 'Software'],
                               'GRP': ['MusicalGRP', 'PublicCORP', 'PrivateCORP', 'AerospaceManufacturer', 'SportsGRP', 'CarManufacturer', 'ORG'],
                               'PER': ['Scientist', 'Artist', 'Athlete', 'Politician', 'Cleric', 'SportsManager', 'OtherPER'],
                               'PROD': ['Clothing', 'Vehicle', 'Food', 'Drink', 'OtherPROD'],
                               'MED': ['Medication/Vaccine', 'MedicalProcedure', 'AnatomicalStructure', 'Symptom', 'Disease']
                               }

COARSE_ENTITY_TYPES = ['LOC', 'CW', 'GRP', 'PER', 'PROD', 'MED']
INSTRUCTION_TEXT = "Ты решаешь задачу NER. Извлеки из текста слова, относящиеся к каждой из следующих сущностей."
COARSE_INSTRUCTION_TEXT = 'Ты решаешь задачу NER. Извлеки из текста слова, относящиеся к каждой из следующих сущностей. LOC, CW, GRP, PER, PROD, MED.'


def fix_typos_in_lables(label: str) -> str:
    if label[-4:] == 'Corp':
        return label[:-4] + "CORP"
    return label


