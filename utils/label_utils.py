import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

def normalize_label(label: str) -> str:
    label = re.sub(r'([a-z])([A-Z])', r'\1 \2', label)
    label = label.replace("_", " ").replace("-", " ")
    return label.strip().lower()

def stemmed_words(text: str) -> set:
    tokens = word_tokenize(text.lower())
    return {stemmer.stem(token) for token in tokens if token.isalpha()}

def explanation_contains_label_stem(expl: str, label: str) -> bool:
    label_stems = {stemmer.stem(w) for w in normalize_label(label).split()}
    expl_stems = stemmed_words(expl)
    return label_stems.issubset(expl_stems)