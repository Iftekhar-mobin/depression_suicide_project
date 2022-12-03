import numpy as np
import pandas as pd
import pickle
from os import path
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from constants import BEST_ENTITIES, CSSR_DIR
from sklearn.preprocessing import LabelEncoder
from utils import create_chord_chart

encoder = LabelEncoder()
scaler = MinMaxScaler()


def get_ngrams(text, ngram_from=2, ngram_to=2, n=None, max_features=20000):
    vec = CountVectorizer(ngram_range=(ngram_from, ngram_to),
                          max_features=max_features,
                          stop_words='english').fit_transform(text)
    bag_of_words = vec.transform(text)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, i]) for word, i in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def generate_count_vecto_bi_gram(corpus):
    data = [corpus.processed[corpus['category'] == 0], corpus.processed[corpus['category'] == 1]]
    collector = []
    for items in data:
        bigrams = get_ngrams(items, ngram_from=2, ngram_to=2, n=30)
        bigrams_df = pd.DataFrame(bigrams)
        bigrams_df.columns = ["Bigram", "Frequency"]
        collector.append(bigrams_df.head(20))
    return collector


def searched_keywords_neighbors_freq(search_neighbor, ngr, feature_name):
    vectorized = CountVectorizer(ngram_range=(ngr, ngr), max_features=20000)
    vectorized.fit_transform(search_neighbor.values)
    features = (vectorized.get_feature_names_out())
    # Applying TFIDF
    vectorized = TfidfVectorizer(ngram_range=(ngr, ngr), max_features=20000)
    X2 = vectorized.fit_transform(search_neighbor.values)
    # Getting top ranking features
    sums = X2.sum(axis=0)
    data1 = []
    for col, term in enumerate(features):
        data1.append((term, sums[0, col]))

    ranking = pd.DataFrame(data1, columns=['term', 'rank'])

    words = (ranking.sort_values('rank', ascending=False))
    words = words.head(10)
    words['feature'] = feature_name

    ranks = np.array(words['rank'])
    ranks = scaler.fit_transform(ranks.reshape(-1, 1))

    words['rank'] = np.concatenate(ranks)
    # collecting 10 best bigrams and their frequency
    return words.head(10)


def make_data_for_chords(df):
    df = df.rename(columns={'term': 'target', 'feature': 'source', 'rank': 'value'})
    labels = df.target.values.tolist() + df.source.values.tolist()
    encoder.fit(labels)
    nodes = encoder.classes_
    encoded_df = pd.DataFrame()
    node_data = pd.DataFrame()
    encoded_df['source'] = encoder.transform(df.source.values.tolist())
    encoded_df['target'] = encoder.transform(df.target.values.tolist())
    encoded_df['value'] = df.value.tolist()
    node_data['nodes'] = nodes
    encoded_df = encoded_df[encoded_df['source'] != encoded_df['target']]
    return encoded_df, node_data


def get_best_entities_network(corpus):
    data = [corpus.processed[corpus['category'] == 0], corpus.processed[corpus['category'] == 1]]
    for dataframe in data:
        searched_results = []
        for searching_words in BEST_ENTITIES:
            searched_results.append(dataframe[dataframe.str.contains(searching_words)])

        i = 0
        collector = pd.DataFrame()
        for results in searched_results:
            collect = searched_keywords_neighbors_freq(results, 2, [BEST_ENTITIES[i]] * 10)
            collector = pd.concat([collector, collect])
            i += 1
        enc_df, node_data = make_data_for_chords(collector)
        create_chord_chart(enc_df, node_data)


def generate_terms_frequency_vector(df):
    pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer())])
    vectorized_data = pipeline.fit_transform(df['processed'])
    n_x_train, n_x_val, n_y_train, n_y_val = train_test_split(vectorized_data, df.code, test_size=0.2, random_state=0)
    return n_x_train, n_x_val, n_y_train, n_y_val, pipeline


def find_best_classifier_optimal_params(df, faster_searching=True):
    n_x_train, n_x_val, n_y_train, n_y_val, pipeline = generate_terms_frequency_vector(df)
    filename = CSSR_DIR + 'finalized_classifier_model.sav'
    # load the model from disk
    if path.exists(filename):
        clf = pickle.load(open(filename, 'rb'))
        return clf, pipeline
    else:
        clf = 0

        if not faster_searching:
            seed = 1
            models = ['GBC', 'RFC', 'KNC', 'SVC', 'logisticRegression']
            classifiers = [
                GradientBoostingClassifier(random_state=seed),
                RandomForestClassifier(random_state=seed, n_jobs=-1),
                KNeighborsClassifier(n_jobs=-1),
                SVC(random_state=seed, probability=True),
                LogisticRegression(solver='newton-cg', multi_class='multinomial')
            ]
            params = {
                models[0]: {'learning_rate': [0.01], 'n_estimators': [100], 'max_depth': [3],
                            'min_samples_split': [2], 'min_samples_leaf': [2]},
                models[1]: {'n_estimators': [100], 'criterion': ['gini'], 'min_samples_split': [2],
                            'min_samples_leaf': [4]},
                models[2]: {'n_neighbors': [5], 'weights': ['distance'], 'leaf_size': [15]},
                models[3]: {'C': [100], 'tol': [0.005], 'gamma': [0.2, 0.4, 0.7, 2], 'degree': [1, 2],
                            'kernel': ['sigmoid', 'rbf']},
                models[4]: {'C': [2000], 'tol': [0.0001]}
            }

            test_scores = []
            for name, estimator in zip(models, classifiers):
                print(name)
                clf = GridSearchCV(estimator, params[name], refit='True', n_jobs=-1, cv=5)
                clf.fit(n_x_train, n_y_train)

                print("best params: " + str(clf.best_params_))
                print("best scores: " + str(clf.best_score_))
                acc = accuracy_score(n_y_val, clf.predict(n_x_val))
                print("Accuracy: {:.4%}".format(acc))
                test_scores.append((acc, clf.best_score_))
        else:
            # if this grid searching technique take ages then do the following
            param_grid = {'kernel': ['sigmoid', 'rbf'], 'gamma': [0.2, 0.4, 0.7, 2], 'degree': [2]}
            clf = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
            clf.fit(n_x_train, n_y_train)

        # save the model to disk
        if clf:
            pickle.dump(clf, open(filename, 'wb'))

        return clf, pipeline
