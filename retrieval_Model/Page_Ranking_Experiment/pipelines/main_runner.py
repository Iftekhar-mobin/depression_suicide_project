from methods_collection import make_question
from pipelines import vectorization
from pipelines import sequence_handler
from pipelines import corpus_handling_methods
import pandas as pd
from pipelines import ranking

# dataset = corpus_handling_methods.corpus_preprocessing(
# "/home/iftekhar/amiebot/Resources/amiebot_dataset/mobicontrol_ver14.csv") dataset.to_csv(
# 'tokenized_cleaned_perpagedata.csv') dataset.head()

# dataset = pd.read_csv("tokenized_cleaned_perpagedata.csv")

dataset = pd.read_csv("../../../Helpers/Title_link_merged_corpus.csv")
dataset = dataset.iloc[:, 2:]
dataset = dataset.rename(columns={"Article": "Data"})

dataset['content'] = dataset.Data

# URL matcher query with the corpus
# for index, col in ques_df.iterrows():
#     for m_index, m_col in dataset.iterrows():
#         if(str(col['URL']) == str(m_col['URL'])):
#             print(col['URL'], m_col['PageID'])
#             break
# cus_ques = pd.read_csv("/home/iftekhar/amiebot/Resources/amiebot_dataset/special_user_query.csv") 35%
# cus_ques = pd.read_csv("/home/iftekhar/amiebot/Resources/amiebot_dataset/user_query.csv") 37%
cus_ques = corpus_handling_methods.query_corpus_processing(
    "/home/iftekhar/amiebot/Resources/amiebot_dataset/support_team_question_pure.csv")
# cus_ques.head()

split_corpus = corpus_handling_methods.corpus_split(dataset, 6)
# split_corpus.head()
perpage_dataset = corpus_handling_methods.corpus_per_page(split_corpus)
# perpage_dataset.head()

sample_size = 30
questions_samples = make_question.question_dataframe_generator_1000(split_corpus, sample_size)
questions_samples.head()

#vec = vectorization.vector_fit_only_tfidf(split_corpus)
#vec = vectorization.bm25_vectorizer_core(split_corpus)
#vec = vectorization.bm25_vectorizer_core(split_corpus)
#vec = vectorization.bm25_vectorizer_exp(split_corpus)
#vec = vectorization.bm25_vectorizer(split_corpus)

vec = vectorization.bm25_vectorizer_cq(dataset)
# print(vocab)
# exit()

MRR_score = 0
sample_count = 0
sum_score = 0
container = []
only_TFIDF = False
for index, col in cus_ques.iterrows():
    print(col['Question'], col['PageID'])
    query = str(col['Question'])
    question_parts = make_question.making_query_collection(query)
    collector = sequence_handler.sequence_searcher(perpage_dataset, question_parts,  col['PageID'], query, True)

    # print(question_parts, collector)
    # exit()

    perpage_sequence_match = sequence_handler.sequence_match_driver(collector, dataset, split_corpus, query, col['PageID'])
    # print(perpage_sequence_match)
    # exit()

    sequence_handler.booster_reporter(perpage_sequence_match, col['Question'], col['PageID'])
    # exit()
    #
    # # ranks = ranking.crude_ranks_bm25(query, split_corpus, vec)
    # #ranks = ranking.crude_ranks_bm25_transformer(query, split_corpus, vec)
    #
    # # exit()
    # #
    # #ranks = ranking.crude_ranks_tfidf('', query, vec, False, True)
    # # ranks = ranking.filtering_ranks(ranks, perpage_sequence_match, query, vec)
    #
    ranks = ranking.crude_ranks_bm25_cdqa(query, vec, perpage_sequence_match)
    #
    print(col['PageID'], "<=>", ranks)

    MRR_score, page_answers, prediction_scores = ranking.MRR_score_collector(ranks, col['PageID'])
    sum_score += MRR_score
    container.append([MRR_score, col['PageID'], page_answers, prediction_scores, col['Question']])
    sample_count += 1

# result = pd.DataFrame(container, columns=['score', 'actual_answer',
# 'page_answers', 'prediction_scores', 'query'])
# result.to_csv('seq_matcher_TFIDF_Vectorizer_performance.csv')
score = sum_score / sample_count
print(score)
