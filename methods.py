import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from constants import PROCESSED_CORPUS, CORPUS_DIR, CSSR_DIR, CSSR_CAT, CSSR_FILES, MODEL_PREDICTION_FILES
from preprocess_cleaning import prepare_classifier_dataset
from sklearn.preprocessing import LabelEncoder
from preprocess_cleaning import clean_process
from keras.layers import TextVectorization
from models import predict_class

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


def get_dataset_for_estimator(dataframe):
    labels = dataframe.code.values
    samples = dataframe.Post.values
    try:
        classes = Le.classes_
    except AttributeError as e:
        dataframe['code'] = Le.fit_transform(dataframe['Label'])
        labels = dataframe.code.values
        classes = Le.classes_

    # Shuffle the data
    seed = 1337
    rng = np.random.RandomState(seed)
    rng.shuffle(samples)
    rng = np.random.RandomState(seed)
    rng.shuffle(labels)

    # Extract a training & validation split
    validation_split = 0.2
    num_validation_samples = int(validation_split * len(samples))
    tr_samples = samples[:-num_validation_samples]
    v_samples = samples[-num_validation_samples:]
    tr_labels = labels[:-num_validation_samples]
    v_labels = labels[-num_validation_samples:]

    vec = TextVectorization(max_tokens=10000, output_sequence_length=500)
    text_ds = tf.data.Dataset.from_tensor_slices(tr_samples).batch(128)
    vec.adapt(text_ds)

    return tr_samples, v_samples, tr_labels, v_labels, vec, classes


def load_dataset():
    file_name = '/home/ifte-home/Documents/mental_health/suicide/CSSRS/500_Reddit_users_posts_labels.csv'
    processed_file = '/home/ifte-home/Documents/mental_health/suicide/CSSRS/500_Reddit_users_posts_labels_processed.csv'
    if not os.path.exists(file_name):
        dataset = prepare_classifier_dataset(pd.read_csv(file_name))
    else:
        dataset = pd.read_csv(processed_file, index_col=0)
    dataset = dataset.dropna()
    return dataset


def aggregate_output_and_count():
    df_total = pd.DataFrame()
    dataset = {}
    for k in range(1, 6):
        file = CSSR_DIR + str(k) + MODEL_PREDICTION_FILES
        dataset[k] = pd.read_csv(file, index_col=0)
        temp = dataset[k]
        dataset[k] = temp[temp['suicide_intensity'] != 'Nan']
        temp = dataset[k][['Processed', 'class', 'suicide_intensity']]
        dataset[k] = temp
        df_total = pd.concat([df_total, dataset[k]])

    return df_total.groupby(['class', 'suicide_intensity']).count().transpose()


def generate_results(dataset, cl_names, mixed_model, vect):
    max_samples = 30000
    if 'text' in dataset.columns:
        dataset = dataset.drop(columns=['text'])

    for x in range(1, 10):
        result = []
        count = 0
        length = len(dataset) - (x * max_samples)
        df_iter = dataset.tail(length)

        for i, j in df_iter.iterrows():
            prediction = predict_class(vect, j['Processed'], cl_names, mixed_model)
            result.append(prediction)
            print(prediction, ",", count)
            count += 1
            if count > max_samples:
                break
            # break

        val = len(df_iter) - len(result)
        df_iter['suicide_intensity'] = result + val * ['Nan']
        df_iter.to_csv(CSSR_DIR + str(x) + '_reddit_dataset_with_CSSR_intensity_glove.csv')


def result_collection():
    items = ['non-suicide', 'suicide']
    dep = ['Depression']
    sui_data = ['Suicide']

    df_total = pd.DataFrame()
    df = {}
    for i in range(1, 10):
        file = CSSR_DIR + str(i) + MODEL_PREDICTION_FILES
        df[i] = pd.read_csv(file, index_col=0)
        temp = df[i]
        df[i] = temp[temp['suicide_intensity'] != 'Nan']
        temp = df[i][['Processed', 'class', 'suicide_intensity']]
        df[i] = temp
        df_total = pd.concat([df_total, df[i]])
    counted_df = df_total.groupby(['class', 'suicide_intensity']).count().transpose()

    for k in items:
        for i in counted_df[k].values:
            for j in i:
                if k == 'suicide':
                    sui_data.append(j)
                else:
                    dep.append(j)

    return dep, sui_data