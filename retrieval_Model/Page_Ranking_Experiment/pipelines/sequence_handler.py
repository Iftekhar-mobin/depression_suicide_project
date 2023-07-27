import re
from pipelines.utility import get_unique_2Dlist
from pipelines.corpus_handling_methods import query_token_remover, single_character_remover, query_features
from methods_collection.make_question import making_query_collection
import spacy
from nltk.corpus import wordnet as wn

nlp = spacy.load("ja_ginza")


def sequence_searcher(corpus_per_page, question_parts, answers, query='', single_token_search=False):
    # Include single word search in the re.findall method
    if single_token_search:
        single_token_query = query.split()
        question_parts = [query] + question_parts + single_token_query
    # print(question_parts)
    # exit()
    collector = query_words_presence(corpus_per_page, question_parts, answers)

    # for debug purpose intentionally making collector null
    # collector = []

    if not collector:
        print("Tokens not found in corpus :/")

        # For debug purpose
        print("Either Token Synonyms are given or Mecab failed to provide Tokens")
        print("Trying to match synonyms with (", query, ")")

        collector = null_matched_handler(corpus_per_page, answers, query)

    return collector


def query_words_presence(corpus, query_list, answers):
    # print(query_list[0])
    # exit()
    bonus = 0
    # collecting IDs having 2/3 length chunks only
    # Query: ウィルス 対策 実施 query_list: ウィルス 対策, 対策 実施
    collector = []
    for index, col in corpus.iterrows():

        for items in query_list:
            point = len(items.split())
            if point > 1:
                bonus = point
            else:
                bonus = 0
            temp = re.findall(r"\b" + items + r"\b", col['Data'])
            if temp:
                collector.append([col['PageID'], temp, len(temp) + len(temp) * bonus])

        # 100 point bonus given for exact match with query
        if str(col['Data']).find(query_list[0]):
            collector.append([col['PageID'], [query_list[0]], 100])

        # for debugging purpose
        # break

    # print(collector)
    # exit()
    # Given high priority where chunks are matched

    chunks_rank = []
    for ids, chunks, point in collector:
        # Provide bonus for the sequence match
        chunks_rank.append([ids, len(chunks) + len(chunks) * bonus])

    collect_ids = []
    collector = []
    for pageID, points in get_unique_2Dlist(chunks_rank):
        collector.append([pageID, sum(points)])
        # for debug purpose only
        collect_ids.append(pageID)
    # print(collector)
    # exit()

    rank = sorted(collector, key=lambda l: l[1], reverse=True)[:int(len(corpus) / 4)]
    # print(rank)
    # exit()

    # for debug purpose only
    print("Step1 Having IDs: ", step_reporter(collect_ids, answers))
    # exit()

    return rank


def null_matched_handler(corpus_per_page, answers, query):
    collected_synonyms = []
    for tokens in query.split():
        collected_synonyms.append(synonym_supplier(tokens))
    collected_synonyms = convert_list_dim(collected_synonyms)
    # print(collected_synonyms)
    # exit()

    collector = []
    for index, col in corpus_per_page.iterrows():
        for synonyms in collected_synonyms:
            temp = re.findall(r"\b" + synonyms + r"\b", col['Data'])
            if temp:
                collector.append([col['PageID'], temp])

    chunks_rank = []
    for ids, matched_tokens in collector:
        chunks_rank.append([ids, len(matched_tokens)])

    collect_ids = []
    collector = []
    for pageID, points in get_unique_2Dlist(chunks_rank):
        collector.append([pageID, sum(points)])
        # for debug purpose only
        collect_ids.append(pageID)

    # print(collector)
    # exit()

    rank = sorted(collector, key=lambda l: l[1], reverse=True)[:int(len(corpus_per_page) / 4)]
    # print(rank)
    # exit()

    # for debug purpose only
    print("Step2 Having IDs: ", step_reporter(collect_ids, answers))
    # exit()

    return rank


def step_reporter(collect_ids, answers):
    return [IDs for IDs in collect_ids if IDs in [int(i) for i in answers.split()]]


def longest_seq_search(query, page_data):
    m = len(query)
    n = len(page_data)
    counter = [[0] * (n + 1) for x in range(m + 1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if query[i] == page_data[j]:
                c = counter[i][j] + 1
                counter[i + 1][j + 1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(query[i - c + 1:i + 1])
                elif c == longest:
                    lcs_set.add(query[i - c + 1:i + 1])

    return lcs_set


def synonym_supplier(string):
    return unique_synonym_list([u.lemma_names('jpn') for u in list(wn.synsets(string, lang='jpn'))])


def unique_synonym_list(syms_terms):
    synonyms = []
    for items in syms_terms:
        for syms in items:
            synonyms.append(syms)
    return list(set(synonyms))


def convert_list_dim(list_item):
    return [x for sublist in list_item for x in sublist]


def check_query_words_at_beginning(collector, dataset, query):
    result = []
    for pages_id in collector:
        page_data = dataset[(dataset['PageID'] == pages_id)].Data.values
        data = ''.join(list(page_data)).split()

        # checking query words present within first lines
        flag = [word for word in query_features(query, nlp) if word in data[0:15]]
        if not flag:
            # print("pages_id is removed, Expecting query string at the beginning: ", pages_id)
            continue
        result.append(pages_id)
    return result


def sequence_match_driver(collected, dataset, split_corpus, query, answers):

    if len(collected) > 20:
        print("User has SEO Tags, e.g. HTag, HTML Title, Link, Bold Text, Highlights, Summary")
        collector = summary_htag_url_matching(dataset, collected, query, answers)
        # print(collector)
        # exit()

        # this one for the corpus generated queries only
        # collector = collected

        # Assumed query words may appear at the beginning of the document.
        # Because usually when we write we repeat the words at the top of the document
        # if collector > 50:
        #     collector = check_query_words_at_beginning(collector, dataset, query)
        # print(collector)
        # exit()

    else:
        collector = collected

    return get_sequence_match_ID(collector, split_corpus, query)


def get_sequence_match_ID(res, split_corpus, query):
    q_parts = making_query_collection(query)
    if len(q_parts) < 2:
        q_parts = query.split()
    match_collection = query_match_collector(res, split_corpus, q_parts)

    if not match_collection:
        match_collection = query_match_collector(res, split_corpus, query.split())
    sorted_list = get_unique_2Dlist(match_collection)

    return sorted_list


def query_match_collector(iterator, corpus, q_parts):
    collector = []
    for ids, score in iterator:
        page_data = corpus[(corpus['PageID'] == ids)].Data.values
        for lines in page_data:
            for parts in q_parts:
                regex = re.compile(r"\b" + parts + r"\b")
                if regex.findall(lines):
                    collector.append([ids, lines])

    return collector


def booster_reporter(sorted_list, Question, PageID):
    answer_ids = str(PageID).split()
    if len(answer_ids) > 1:
        flag = False
        num_ans = len(answer_ids)
        for items in answer_ids:
            ids_list = []
            end_tag = len(sorted_list)
            for ids, chunks in sorted_list:
                ids_list.append(ids)
                if ids == int(items):
                    print("Booster having IDs", Question, items)
                    if not flag:
                        saver = []
                        for collected_ids, collected_text in sorted_list:
                            saver.append(collected_ids)
                        print('collected ids list: ', saver)
                        flag = True
                    break
                elif end_tag < 2 and num_ans < 2:
                    print("failed to get ID", Question, items)
                    return ids_list
                end_tag -= 1
            num_ans -= 1
    else:
        chunks_id_collectors(sorted_list, Question, PageID)


def chunks_id_collectors(sorted_list, Question, PageID):
    ids_list = []
    end_tag = len(sorted_list)
    for ids, items in sorted_list:
        ids_list.append(ids)
        if ids == int(PageID):
            print("Booster having IDs", Question, PageID)
            break
        elif end_tag < 2:
            print("failed to get ID", Question, PageID)
            print(ids_list)
        end_tag -= 1


def summary_htag_url_matching(dataset, collected, query, answers):
    if 'Summary' in list(dataset.columns):
        # Debug purpose
        # print("Summary column found")
        saver = []
        for index, col in dataset.iterrows():
            for tokens in query.split():
                temp = re.findall(tokens, col['Summary'])
                if temp:
                    saver.append([col['PageID'], len(temp)])
        # print(collector)
        # exit()

        collector = []
        for pageID, points in get_unique_2Dlist(saver):
            collector.append([pageID, sum(points)])
        # print(collector)
        # exit()

        collect_ids = []
        matched = []
        for pageID, points in collector:
            for IDs, scores in collected:
                if IDs == pageID:
                    matched.append([pageID, points])
                    # for debug purpose only
                    collect_ids.append(pageID)

        if len(matched) < 10:
            return collected
        else:
            rank = sorted(matched, key=lambda l: l[1], reverse=True)[:int(len(dataset) / 4)]
            # print(rank)

            # for debug purpose only
            print("Step3 Having IDs: ", step_reporter(collect_ids, answers))
            # exit()
            return rank
    else:
        return collected
