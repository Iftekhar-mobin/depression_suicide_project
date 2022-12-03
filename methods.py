import pandas as pd
import random
from constants import PROCESSED_CORPUS, CORPUS_DIR, CSSR_DIR, CSSR_CAT, CSSR_FILES
from sklearn.preprocessing import LabelEncoder
from preprocess_cleaning import clean_process

Le = LabelEncoder()


def generate_samples(categories, res):
    data = []
    category = []
    for p in res:
        data.append(''.join(p))
        category.append(categories)
    return pd.DataFrame(zip(data, category), columns=['Post', 'Label'])


def generate_classifier_training_dataset(df):
    sentence_num = 10
    index = 0
    dataset = pd.DataFrame()

    for items in CSSR_FILES:
        frame = pd.read_csv(CSSR_DIR + items)
        col_name = list(frame.columns)

        # Generating some synthetic samples
        # making samples such a way that category becomes equally distributed
        if index == 0:
            size = 4
        elif index == 1:
            size = 52
        elif index == 3:
            size = 14
        else:
            size = 45
        for k in range(size):
            temp = [col_name[i:i + sentence_num] for i in range(0, len(col_name), sentence_num)]
            dataset = pd.concat([dataset, generate_samples(CSSR_CAT[index], temp)])
            random.shuffle(col_name)
        index += 1

    dataset = dataset.sample(frac=1)
    df = pd.concat([df, dataset])
    df = df[df['Label'] != 'Supportive']
    df['processed'] = df.Post.apply(lambda x: clean_process(x))
    df['code'] = Le.fit_transform(df['Label'])

    return df


def get_count(pipeline, clf):
    reddit_dataset = pd.read_csv(CORPUS_DIR+PROCESSED_CORPUS)
    vectorized_data_red = pipeline.transform(reddit_dataset['processed'])
    result = clf.predict(vectorized_data_red)
    reddit_dataset['suicide_intensity'] = result
    reddit_dataset['intensity'] = list(Le.inverse_transform(result))
    reddit_dataset.to_csv(CSSR_DIR + 'reddit_dataset_with_CSSR_intensity.csv')
    result = reddit_dataset.groupby(['class', 'intensity']).count()
    return result.transpose()
