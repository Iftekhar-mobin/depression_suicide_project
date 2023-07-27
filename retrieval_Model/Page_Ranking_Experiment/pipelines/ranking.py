from sklearn.metrics.pairwise import cosine_similarity
from pipelines.visualization import visualizer
from pipelines.sequence_handler import convert_list_dim
import numpy as np
import MeCab
from yellowbrick.text.tsne import tsne

from pipelines import vectorization

mt = MeCab.Tagger('')
mt.parse('')
from methods_collection.make_question import making_query_collection
from pipelines import utility


def whole_corpus_mapping(corpus_scores):
    idx_scores = [(idx, score) for idx, score in enumerate(corpus_scores)]
    rank = sorted(idx_scores, key=(lambda tup: tup[1]), reverse=True)[: 30]
    collector = []
    for ids, score in rank:
        collector.append([ids, score[0]])
    visualizer(collector)


def crude_ranks_bm25_cdqa(query, vec, booster_result):
    # corpus_scores = bm25_matrix.dot(question_vector.T).toarray()
    # query_vec = vectorizer.transform(query)
    # Transform a count matrix to a tf or tf-idf representation
    # array([[1, 1, 1, 1, 0, 1, 0, 0],
    #        [1, 2, 0, 1, 1, 1, 0, 0],
    #        [1, 0, 0, 1, 0, 1, 1, 1],
    #        [1, 1, 1, 1, 0, 1, 0, 0]])

    corpus_scores, query_vec = vec.predict([query])

    print(vec.get_feature_names())
    exit()

    #print(corpus_scores, query_vec.toarray())
    # exit()

    # To visualize the ranks
    # whole_corpus_mapping(corpus_scores)
    # exit()

    rank = []
    ids_list = []
    double_rank = []
    for ids, items in booster_result:
        dummy, collected_lines_vec = vec.predict([' '.join(items)])
        ids_list.append(ids)
        rank.append(cosine_similarity(query_vec, collected_lines_vec))
        collector = []
        for chunk in items:
            collector.append(chunk)
        dummy, chunks_vec = vec.predict(collector)
        double_rank.append(cosine_similarity(query_vec, chunks_vec))

    print("joined Rank", rank, "Chunk Rank", double_rank)
    exit()

    summary_store = []
    for ids, score in zip(ids_list, convert_list_dim(rank)):
        summary_store.append([ids, score[0]])
    # print(summary_store)

    ranks = sorted(summary_store, key=lambda l: l[1], reverse=True)
    # print(ranks)
    visualizer(ranks)
    exit()

    # saver = []
    # for items in double_rank:
    #     item_1d = convert_list_dim(items)
    #     x = [x for x in item_1d if x > 0]
    #     for scores in x:
    #
    #         saver.append(x)
    # print(saver)
    # exit()
    # print(double_rank)
    #
    # exit()

    #ranks = sorted(zip(ids_list, convert_list_dim(double_rank)), key=lambda l: l[1], reverse=True)[:3]
    # print(d_ranks)
    # exit()
    #

    return ranks


def crude_ranks_bm25(query, split_corpus, vec):
    # IDs_list = []
    # seq_corpus_collection = []
    # for ids, items in sorted_list:
    #     seq_corpus_collection.append(items)
    #     IDs_list.append(ids)
    # # print(corpus_collection)
    # corpus = [x for sublist in seq_corpus_collection for x in sublist]
    #
    # seq_lines_collector = []
    # for items in corpus:
    #     seq_lines_collector.append(items.split())
    # # print(collector)
    # # exit()

    scores = vec.search([word for word in query.lower().split()])
    collector = []
    for score, IDs in zip(scores, split_corpus.PageID.values.tolist()):
        collector.append([IDs, score])

    total_ranks = []
    for Ids, ranks in utility.get_unique_2Dlist(collector):
        adder = 0
        for rank in ranks:
            adder += rank
        total_ranks.append([Ids, adder])

    ranks_bm25 = sorted(total_ranks, key=lambda l: l[1], reverse=True)[:3]
    # print(ranks_bm25)
    # exit()
    # collect_Ids = []
    # for Ids, ranks in ranks_bm25:
    #     collect_Ids.append(Ids)
    #
    # common_ids = [terms for terms in IDs_list if terms in collect_Ids]
    #
    # # print(IDs_list, collect_Ids, common_ids)
    # # exit()
    # print(sorted(common_ids, reverse=True)[:3])
    # exit()

    # for score, doc in zip(scores, corpus):
    #     score = round(score, 3)
    #     print(str(score) + '\t' + doc)

    return ranks_bm25


def crude_ranks_bm25_transformer(query, split_corpus, vec, extra_ranks=False):
    question_parts = making_query_collection(query)
    if len(question_parts) < 2:
        question_parts = query.split()

    scores = vec.transform(query.lower(), split_corpus.Data.values.tolist())
    # print(query_vec)

    collector = []
    for score, IDs in zip(scores, split_corpus.PageID.values.tolist()):
        collector.append([IDs, score])
    # print(collector)
    # exit()
    total_ranks = []
    for Ids, ranks in utility.get_unique_2Dlist(collector):
        adder = 0
        for rank in ranks:
            adder += rank
        total_ranks.append([Ids, adder])
    # print(total_ranks)

    ranks_bm25 = sorted(total_ranks, key=lambda l: l[1], reverse=True)[:3]
    # print(ranks_bm25)

    return ranks_bm25


def crude_ranks_tfidf(sorted_list, query, vector, extra_ranks=False, tfidf=False):
    if tfidf:
        query_vec = vector.transform([query])
        print(query_vec, "Hi")
        exit()
        # rank = []
        # ids_list = []
        # for ids, items in sorted_list:
        #     collected_lines_vec = vector.transform([' '.join(items)])
        #     ids_list.append(ids)
        #     rank.append(cosine_similarity(query_vec, collected_lines_vec))
    else:
        question_parts = making_query_collection(query)
        if len(question_parts) < 2:
            question_parts = query.split()

        query_vec = vector.transform([query.lower()])
        rank = []
        ids_list = []
        for ids, items in sorted_list:
            collected_lines_vec = vector.transform([' '.join(items)])
            ids_list.append(ids)
            rank.append(cosine_similarity(query_vec, collected_lines_vec))
        # print(rank)

        flat = [x for sublist in rank for x in sublist]
        if not extra_ranks:
            ranks = sorted(zip(ids_list, flat), key=lambda l: l[1], reverse=True)[:3]
        else:
            ranks = sorted(zip(ids_list, flat), key=lambda l: l[1], reverse=True)[:10]

    return ranks


def delete_duplicate_ids(ranks):
    temp = 0
    ids_list = []
    filtered = []
    for page_id, score in ranks:
        if page_id == temp:
            continue
        else:
            ids_list.append(page_id)
            filtered.append([page_id, score[0]])
            temp = page_id
    return filtered, ids_list


def filtering_ranks(ranks, sorted_list, query, vector):
    filtered, ids_list = delete_duplicate_ids(ranks)
    # print("deleted duplicate: ", filtered, ids_list)
    saved_list = sorted_list
    if len(ids_list) < 3:
        for ids in ids_list:
            index = 0
            for matched_id, items in sorted_list:
                if ids == matched_id:
                    saved_list.pop(index)
                index += 1
        # print("Saved list", saved_list)
        # exit()
        extra_ranks = True
        extra_ranks = crude_ranks_tfidf(saved_list, query, vector, extra_ranks)
        filtered_extra, ids_list_extra = delete_duplicate_ids(extra_ranks)
        sum_filtered = filtered + filtered_extra
        return sorted(sum_filtered, key=lambda l: l[1], reverse=True)[:3]
    else:
        return filtered


def get_vector(text, gensim_model):
    sum_vec = np.zeros(200)
    word_count = 0
    node = mt.parseToNode(text)
    while node:
        fields = node.feature.split(",")
        if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
            try:
                temp = gensim_model.wv[node.surface]
            except KeyError:
                temp = 0
            sum_vec += temp
            word_count += 1
        node = node.next
    return sum_vec / word_count


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def word2vec_ranks(perpage_sequence_match, query, gensim_model):
    X = get_vector(query, gensim_model)
    rank = []
    ids_list = []
    for ids, items in perpage_sequence_match:
        for sentences in items:
            Y = get_vector(sentences, gensim_model)
            ids_list.append(ids)
            rank.append(cos_sim(X, Y))
    return sorted(zip(ids_list, rank), key=lambda l: l[1], reverse=True)[:3]


def MRR_score_collector(ranks, actual_answers):
    predicted_answers = []
    prediction_scores = []
    for ids, score in ranks:
        predicted_answers.append(ids)
        prediction_scores.append(score)

    actual_answers_ids = str(actual_answers).split()
    if len(actual_answers_ids) > 1:
        saver = 0
        for actual_answers_id in actual_answers_ids:
            temp = 0
            for predicted_ids in predicted_answers:
                if actual_answers_id == str(predicted_ids):
                    MRR_score = mean_reciprocal_rank_score(int(actual_answers_id), predicted_answers)
                    if MRR_score > temp:
                        temp = MRR_score
            MRR_score = temp
            if MRR_score > saver:
                saver = MRR_score
        MRR_score = saver

    else:
        MRR_score = mean_reciprocal_rank_score(int(actual_answers), predicted_answers)

    print("MRR: ", MRR_score, 'answer_ids', actual_answers_ids, 'page_answers', predicted_answers)

    return MRR_score, predicted_answers, prediction_scores


def mean_reciprocal_rank_score(actual_value, predicted_values):
    pos = 0
    val = 0
    for i in predicted_values:
        if i == actual_value and pos == 0:
            val = 1
            break
        elif i == actual_value and pos == 1:
            val = 0.5
            break
        elif i == actual_value and pos == 2:
            val = 0.33
            break
        else:
            val = 0
        pos += 1

    return val
