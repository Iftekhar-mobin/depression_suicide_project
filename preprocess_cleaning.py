import pandas as pd
import random
import re
import nltk
import string
from os import path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from constants import CORPUS_PATH, CORPUS_DIR, PROCESSED_CORPUS, CORPUS_EQUAL_CLASS, CSSR_DIR, CSSR_FILES, CSSR_CAT, \
    NOISE

LE = LabelEncoder()
lemme = WordNetLemmatizer()
stop_words_nltk = set(stopwords.words('english'))
remove_punc_dict = dict((ord(pun), ' ') for pun in string.punctuation)



def generate_equal_class_corpus():
    corpus = pd.read_csv(CORPUS_DIR+PROCESSED_CORPUS)
    temp = corpus[corpus['processed'].map(len) > 300]
    dep_data = temp[temp['category'] == '0']
    temp = corpus[(corpus['processed'].map(len) > 1000)]
    sui_data = temp[temp['category'] == '1']
    data = pd.concat([dep_data, sui_data])
    data = data.sample(frac=1)
    data['processed'] = data.processed.apply(lambda x: ' '.join([w for w in str(x).split() if w not in NOISE]))
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
    LE.fit(labels)

    encoded_df = pd.DataFrame()
    enc_source = LE.transform(collector.source.values.tolist())
    enc_target = LE.transform(collector.target.values.tolist())

    encoded_df['source'] = enc_source
    encoded_df['target'] = enc_target
    encoded_df['value'] = collector.value.tolist()

    nodes = LE.classes_
    node_data = pd.DataFrame()
    node_data['nodes'] = nodes
    encoded_df = encoded_df[encoded_df['source'] != encoded_df['target']]
    return node_data, encoded_df


def prepare_lines_from_df(frame):
    data = {'Post': [], 'Label': []}
    for _, col in frame.iterrows():
        for items in col['Post'].split(','):
            data['Post'].append(items)
            data['Label'].append(col['Label'])
    return pd.DataFrame(data)

def generate_dataset(classes, res):
    data = []
    category = []
    for sample in res:
        data.append(''.join(sample))
        category.append(classes)
    return pd.DataFrame(zip(data, category), columns=['Post', 'Label'])

def generate_samples():
    sentence_num = 10
    index = 0
    dataset = pd.DataFrame()

    for items in CSSR_FILES:
        frame = pd.read_csv(CSSR_DIR + items, index_col=0)
        col_name = list(frame.columns)
        # Making the class evenly distributed
        if index == 0:
            size = 4
        elif index == 1:
            size = 52
        elif index == 3:
            size = 14
        else:
            size = 45
        for k in range(size):
            temp = [col_name[items:items + sentence_num] for items in range(0, len(col_name), sentence_num)]
            dataset = pd.concat([dataset, generate_dataset(CSSR_CAT[index], temp)])
            random.shuffle(col_name)
        index += 1
    return dataset.sample(frac=1)


def process(text):
    text = text.lower()
    text = re.sub('\W+', ' ', text)
    text = [x for x in [lemme.lemmatize(w) for w in text.split()] if x not in stop_words_nltk]
    return ' '.join(text)


def prepare_classifier_dataset(dataset):
    dataframe = pd.concat([prepare_lines_from_df(dataset), generate_samples()])
    dataframe['code'] = LE.fit_transform(dataframe['Label'])
    dataframe['Post'] = dataframe.Post.apply(lambda sample: process(sample))
    dataframe = dataframe.dropna()
    dataframe.to_csv(
        '/home/ifte-home/Documents/mental_health/suicide/CSSRS/500_Reddit_users_posts_labels_processed.csv')
    return dataframe


def prepare_dataset_for_predict():
    file_name = CORPUS_DIR + PROCESSED_CORPUS
    if not path.exists(file_name):
        dataframe = pd.read_csv(CORPUS_PATH, index_col=0)
        dataframe['Processed'] = dataframe.text.apply(lambda data: process(data))
        dataframe.to_csv(file_name)
        dataframe = dataframe.dropna()
        dataframe["Processed"] = dataframe["Processed"].astype(str)
    else:
        dataframe = pd.read_csv(file_name)
    return dataframe.dropna()
