import pandas as pd
import sys
import MeCab
import re
import random
from pathlib import Path
from urllib.parse import urlparse
import methods_collection.make_question as question_maker

mecab = MeCab.Tagger('-Owakati')


# def corpus_preprocessing(dataset):
#     dataset = pd.read_csv(dataset)
#     dataset = dataset.rename(columns={"text": "Data", "page": "PageID"})
#     dataset = extract_link_merge_title(dataset)
#     dataset = corpus_merger(dataset)
#     print(dataset.head())
#     exit()
#     dataset.Data = dataset.Data.apply(lambda x: mecab_tokenization(x))
#     dataset.Data = dataset.Data.apply(lambda x: cleaner(x))
#     dataset.Data = dataset.Data.apply(lambda x: single_character_remover(x))
#     dataset.Summary = dataset.Summary.apply(lambda x: mecab_tokenization(x))
#     dataset.Summary = dataset.Summary.apply(lambda x: cleaner(x))
#     dataset.Summary = dataset.Summary.apply(lambda x: single_character_remover(x))
#     return dataset

def generate_vocabulary(dataset):
    dataset['Data'] = dataset.Data.apply(lambda x: single_character_remover(x))
    vocab = []
    for index, col in dataset.iterrows():
        vocab.append(col['Data'].split())

    vocabulary = list(set([v for sublist in vocab for v in sublist]))

def query_features(query, nlp):
    temp = []
    for items in query.split():
        doc = nlp(items)
        for ent in doc.ents:
            if ent.text:
                temp.append(ent.text)
        for np in doc.noun_chunks:
            temp.append(str(np))

    return list(set(temp))


def query_corpus_processing(corpus):
    cus_ques = pd.read_csv(corpus)
    cus_ques.Question = cus_ques.Question.apply(lambda x: mecab_tokenization(x))
    cus_ques.Question = cus_ques.Question.apply(lambda x: single_character_remover(x))
    cus_ques.Question = cus_ques.Question.apply(lambda x: cleaner(x))
    return cus_ques


def get_stop_word_ja():
    stop_word_file = Path("/home/iftekhar/AI-system/Helpers/stop_word_ja.txt")
    with open(stop_word_file, encoding='utf-8') as f:
        stop_word_list = f.read().splitlines()
    return stop_word_list


def mecab_tokenization(text):
    q = mecab.parse(text)
    q_parts = q.split()

    return ' '.join([word for word in q_parts if not word in get_stop_word_ja()])


def single_character_remover(text):
    collector = []
    for items in text.split():
        if len(items) < 2:
            replaced = re.sub(r'[ぁ-んァ-ン]', '', items)
            replaced = re.sub(r'[A-Za-z]', '', replaced)
            replaced = re.sub(r'[0-9]', '', replaced)
            collector.append(replaced)
        else:
            collector.append(items)

    return ' '.join([temp.strip(' ') for temp in collector])


# def corpus_merger(corpus):
#     total_pages = corpus.PageID.unique()
#     data = PageID = url = title = summary = []
#     for i in list(total_pages):
#         page_data = corpus[corpus.PageID == i].Data.values
#         data.append(' '.join(list(page_data)))
#         PageID.append(i)
#         u = corpus[corpus.PageID == i].url.values
#         url.append(' '.join(list(set(u))))
#         t = corpus[corpus.PageID == i].title.values
#         title.append(' '.join(list(set(t))))
#         s = corpus[corpus.PageID == i].Summary.values
#         summary.append(' '.join(list(set(s))))
#
#     df = pd.DataFrame(zip(PageID, data, summary, url, title), columns=["PageID", "Data", "Summary", "URL", "Title"])
#     return df


def corpus_per_page(corpus):
    total_pages = corpus.PageID.unique()
    data = []
    PageID = []
    for i in list(total_pages):
        page_data = corpus[corpus.PageID == i].Data.values
        data.append(' '.join(list(page_data)))
        PageID.append(i)
    df = pd.DataFrame(zip(data, PageID), columns=["Data", "PageID"])
    return df


def corpus_split(corpus, sentence_length):
    labels = corpus.PageID.unique()
    lines = []
    all_ids = []
    for i in list(labels):
        text_list = corpus[corpus.PageID == i].Data.values
        split_text = fixed_length_sentence(' '.join(text_list), sentence_length)
        ids = [i] * len(split_text)
        lines += split_text
        all_ids += ids
    split_corpus = pd.DataFrame(zip(lines, all_ids), columns=["Data", "PageID"])
    return split_corpus


def cleaner(text):
    collector = []
    for items in text.split():
        cleaned = clean_text(items)
        cleaned = re.sub(r"\s+", '', cleaned)
        if cleaned is not '' or cleaned is not ' ':
            collector.append(clean_text(items))

    return ' '.join(collector)


def clean_text(text):
    replaced = text.replace("\\", "")
    replaced = replaced.replace("+", "")
    replaced = re.sub('_', '', replaced)
    replaced = re.sub('\W+', ' ', replaced)
    replaced = re.sub(r'￥', '', replaced)  # 【】の除去
    replaced = re.sub(r'．', '', replaced)  # ・ の除去
    replaced = re.sub(r'｣', '', replaced)  # （）の除去
    replaced = re.sub(r'｢', '', replaced)  # ［］の除去
    replaced = re.sub(r'～', '', replaced)  # メンションの除去
    replaced = re.sub(r'｜', '', replaced)  # URLの除去
    replaced = re.sub(r'＠', '', replaced)  # 全角空白の除去
    replaced = re.sub(r'？', '', replaced)  # 数字の除去
    replaced = re.sub(r'％', '', replaced)
    replaced = re.sub(r'＝', '', replaced)
    replaced = re.sub(r'！', '', replaced)
    replaced = re.sub(r'｝', '', replaced)
    replaced = re.sub(r'：', '', replaced)
    replaced = re.sub(r'－', '', replaced)
    replaced = re.sub(r'･', '', replaced)
    replaced = re.sub(r'ｔ', '', replaced)
    replaced = re.sub(r'ｋ', '', replaced)
    replaced = re.sub(r'ｄ', '', replaced)
    replaced = re.sub(r'\d+', '', replaced)

    return replaced


def fixed_length_sentence(contents, word_limit):
    contents_list = contents.split()
    end = len(contents_list)
    count = 0
    collector = []
    line = []
    for items in contents_list:
        if count < word_limit - 1 and end > 1:
            collector.append(items)
            count += 1
        else:
            collector.append(items)
            line.append(' '.join(collector))
            collector = []
            count = 0
        end -= 1
    return line


def page_text_split(page_text, word_limit):
    page_text = page_text.split()
    chunks = [' '.join(page_text[i:i + word_limit]) for i in range(0,
                                                                   len(page_text), word_limit)]
    return chunks


def query_token_remover(query):
    query_list = query.split()
    query_list = [items for items in query_list if items not in question_maker.get_questions_delimiter_ja()]
    query_text = ' '.join(query_list)
    return query_text


def deep_clean(matched_sequence):
    rank = []
    collector = []
    ids_list = []
    for ids, items in matched_sequence:
        for sentences in items:
            clean_sentence = []
            for items in ''.join(sentences).split():
                if len(items) < 2:
                    items = re.sub(r'[ぁ-ん]', '', items)
                    items = re.sub(r'[ア-ン]', '', items)
                    items = re.sub(r'[A-Za-z]', '', items)
                    clean_sentence.append(items)
                else:
                    clean_sentence.append(items)
            cleaned_text = [cleaned for cleaned in clean_sentence if cleaned not in ['']]
            collector.append([ids, ' '.join(cleaned_text)])
    return collector


def extract_link_merge_title(corpus):
    link_text = []
    for index, col in corpus.iterrows():
        disassembled = urlparse(col['url'])
        only_url = re.sub('/products/help/v14/', '', disassembled.path)
        only_url = re.sub('.html', '', only_url)
        only_url = re.sub('_', ' ', only_url)
        only_url = re.sub('/', ' ', only_url)
        link_text.append(only_url)

    corpus['parsed_link'] = link_text
    corpus['Summary'] = corpus.title + " " + corpus.parsed_link

    return corpus
