import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from constants import CORPUS_PATH, CORPUS_DIR, PROCESSED_CORPUS, CORPUS_EQUAL_CLASS

LE = LabelEncoder()
lemme = WordNetLemmatizer()
stop_words_nltk = set(stopwords.words('english'))
remove_punc_dict = dict((ord(pun), ' ') for pun in string.punctuation)
noise = ['http', 'na', 'paul', 'jake', 'gon', 'filler', 'pop', 'got', 'ha', 'ni', 'lt', 'gt', 'wa', 'ur', 'cum',
         'would', 'sus', 'cheese', 'get', 'cecil', 'want', ]


def generate_equal_class_corpus():
    corpus = pd.read_csv(CORPUS_DIR+PROCESSED_CORPUS)
    temp = corpus[corpus['processed'].map(len) > 300]
    dep_data = temp[temp['category'] == '0']
    temp = corpus[(corpus['processed'].map(len) > 1000)]
    sui_data = temp[temp['category'] == '1']
    data = pd.concat([dep_data, sui_data])
    data = data.sample(frac=1)
    data['processed'] = data.processed.apply(lambda x: ' '.join([w for w in str(x).split() if w not in noise]))
    data.to_csv(CORPUS_DIR + CORPUS_EQUAL_CLASS)
    return data


def lem_tokens(tokens):
    return [w for w in [lemme.lemmatize(token) for token in tokens] if not w in stop_words_nltk]


def lem_normalization(text):
    #     text = process(text)
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))


def clean_process(text):
    text = re.sub('\W+', ' ', text)
    text = re.sub(r'[0-9]', ' ', text)
    x = lem_normalization(str(text))
    text = ' '.join(x)
    collect_text = []

    for items in text.split():
        if len(items) > 1:
            collect_text.append(items)
    return ' '.join(collect_text)


# Load dataset and clean dataset
def dataset_load_clean_preprocess():
    corpus = pd.read_csv(CORPUS_PATH)
    corpus['processed'] = corpus.text.apply(lambda x: clean_process(x))
    corpus = corpus.dropna()
    corpus['category'] = LE.fit_transform(corpus['class'])
    corpus.to_csv(CORPUS_DIR + PROCESSED_CORPUS)
    return corpus

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(1,100))

def prepare_chord_chart_data(collector):
    collector = collector.rename(columns={'term': 'target', 'feature': 'source', 'rank': 'value'})
    labels = collector.target.values.tolist() + collector.source.values.tolist()
    encoder.fit(labels)

    encoded_df = pd.DataFrame()

    enc_source = encoder.transform(collector.source.values.tolist())
    enc_target = encoder.transform(collector.target.values.tolist())

    encoded_df['source'] = enc_source
    encoded_df['target'] = enc_target
    encoded_df['value'] = collector.value.tolist()

    nodes = encoder.classes_
    node_data = pd.DataFrame()
    node_data['nodes'] = nodes
    encoded_df = encoded_df[encoded_df['source'] != encoded_df['target']]
    return node_data, encoded_df