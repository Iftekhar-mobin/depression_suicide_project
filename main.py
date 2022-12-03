import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from preprocess_cleaning import dataset_load_clean_preprocess, generate_equal_class_corpus
from models import generate_count_vecto_bi_gram, get_best_entities_network, find_best_classifier_optimal_params
from utils import create_hor_bar_plot, create_scatter_plot, create_chord_chart, prepare_risk_count
from constants import CORPUS_DIR, PROCESSED_CORPUS, PICKLED_CORPUS, SCATTERTEXT_FILE, CSSR_DIR, CSSR_DATASET
from methods import generate_classifier_training_dataset, get_count
import scattertext as st


def doc_length_analysis(dataframe):
    data = {'Depression': [], 'Suicide': []}
    ind = []
    dataframe['processed'] = dataframe['processed'].astype(str)
    for i in range(100, 1500, 100):
        temp = dataframe[dataframe['processed'].map(len) > i]
        x = temp.category.value_counts()
        data['Depression'].append(x[0])
        data['Suicide'].append(x[1])
        ind.append(i)

    return pd.DataFrame(data, index=ind)


def load_data(corpus_file, already_clean=True):
    if not already_clean:
        return dataset_load_clean_preprocess()
    else:
        corpus_df = pd.read_csv(corpus_file)
        corpus_df = corpus_df.dropna()
        corpus_df['processed'] = corpus_df['processed'].astype(str)
        return corpus_df, corpus_df[corpus_df['processed'].map(len) > 1000]


def create_terms_frequency_rank(corpus_df):
    title = ['Frequency Rank of terms in Suicide class', 'Frequency Rank of terms in Depression class']
    for i in range(2):
        cv = CountVectorizer()
        cv.fit(corpus_df['processed'])

        created_df = pd.DataFrame(cv.transform(corpus_df.processed[corpus_df['category'] == i]).todense(),
                                  columns=cv.get_feature_names())
        total_words = created_df.sum(axis=0)

        top_words = total_words.sort_values(ascending=False).head(25)
        top_df = pd.DataFrame(top_words, columns=["count"])
        create_hor_bar_plot(top_df, title[i])


def generate_scatter_text_chart(scatter_data):
    file_path = CORPUS_DIR + PICKLED_CORPUS
    html_file_path = CORPUS_DIR + SCATTERTEXT_FILE
    nlp = st.whitespace_nlp_with_sentences
    scatter_data['parsed'] = scatter_data.processed.apply(nlp)
    scatter_data.to_pickle(file_path)
    scatter_data = pd.read_pickle(file_path)
    corpus = st.CorpusFromParsedDocuments(scatter_data, category_col="class",
                                          parsed_col="parsed").build().get_unigram_corpus().compact(
        st.AssociationCompactor(500))

    create_scatter_plot(scatter_data, corpus, html_file_path)


# if already cleaned
df, short_df = load_data(CORPUS_DIR + PROCESSED_CORPUS, True)

# create_bar_plot(doc_length_analysis(df), "Document Length", "Frequency",
# "Category frequency based on sentence length")
# generate frequency based charts
# create_terms_frequency_rank(short_df)
generate_scatter_text_chart(short_df)
df_eq_class = generate_equal_class_corpus()
bi_gram_result = generate_count_vecto_bi_gram(df_eq_class)
get_best_entities_network(df_eq_class)


############################
training_data = generate_classifier_training_dataset(pd.read_csv(CSSR_DIR + CSSR_DATASET, encoding="ISO-8859-1"))
classifier, pipe = find_best_classifier_optimal_params(training_data, faster_searching=True)
count_risk = get_count(pipe, classifier)
similarity_score = prepare_risk_count()
