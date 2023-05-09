import numpy as np

TAGS = ['ACTIVITY',
        'ADMINISTRATION_ROUTE',
        'ANATOMY',
        'CHEM',
        'DEVICE',
        'DISO',
        'FINDING',
        'FOOD',
        'GENE',
        'INJURY_POISONING',
        'HEALTH_CARE_ACTIVITY',
        'LABPROC',
        'LIVB',
        'MEDPROC',
        'MENTALPROC',
        'PHYS',
        'SCIPROC',
        'AGE',
        'CITY',
        'COUNTRY',
        'DATE',
        'DISTRICT',
        'EVENT',
        'FAMILY',
        'FACILITY',
        'LOCATION',
        'MONEY',
        'NATIONALITY',
        'NUMBER',
        'ORDINAL',
        'ORGANIZATION',
        'PERCENT',
        'PERSON',
        'PRODUCT',
        'PROFESSION',
        'STATE_OR_PROVINCE',
        'TIME',
        'LANGUAGE',
        'CRIME',
        'AWARD'
        ]


def get_tagset(tagging_scheme: str = "BIO"):
    # create pairs B-tag and I-tag from fine-grainder tagset of NerelBIO
    if tagging_scheme == "BIO":
        iob_tags = ['O'] + list(np.array([[f'B-{tag}', f'I-{tag}'] for tag in TAGS]).flatten())
    elif tagging_scheme == "BILOU":
        raise NotImplementedError
    return dict(zip(iob_tags, range(len(iob_tags))))
