from typing import Tuple
import streamlit as st
import spacy_streamlit
from PIL import Image
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

MODEL_PATH = 'cointegrated/rubert-tiny2' 
LOGO_PATH = 'medner_logo_v1.png'
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

hugginface_pipeline = pipeline(model=model,
                               tokenizer=tokenizer,
                               task='ner', 
                               aggregation_strategy='average')

def generate_doc(text: str) -> Tuple[dict, list[dict]]:
    prediction = hugginface_pipeline(text)
    ents = []
    for entity in prediction:
                ents.append({
                    'start': entity['start'],
                    'end': entity['end'],
                    'label': 'Symptom'
                })    
    return {'text': text, 'ents': ents}, prediction


def main():
    colors = {"symptom" :"linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
    options = {"colors": colors}
    our_image = Image.open(LOGO_PATH)
    st.image(our_image)
    
    raw_text = st.text_area("Your Text", placeholder="Enter Text Here")
    if len(raw_text) != 0:
        doc, prediction = generate_doc(raw_text)
        spacy_streamlit.visualize_ner([doc], labels=['symptom'], manual=True, show_table=False, displacy_options=options)
        st.caption('Raw prediction:')
        st.json(prediction, expanded=False)


if __name__ == '__main__':
    main()
    