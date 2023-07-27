from pipelines.BM25_Exps.bm25 import BM25
#from pipelines.bm25_core.bm25_retriever import BM25Retriever
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pipelines.BM25_Exps.BM25_1 import BM25 as BM25_Exp1
from pipelines.BM25_Exps.cdQA.cdqa.retriever.retriever_sklearn import BM25Retriever


vectorizer = CountVectorizer()
TFIDF_vectorizer = TfidfVectorizer()


def vector_fit_only_tfidf(split_corpus):
    vec = TFIDF_vectorizer.fit(split_corpus.Data.values.tolist())
    return vec


def bm25_vectorizer(split_corpus):
    bm25 = BM25()
    bm25.fit(split_corpus.Data.values.tolist())
    return bm25


# def bm25_vectorizer_core(split_corpus):
#     bm25 = BM25Retriever()
#     vec = bm25.fit_vectorizer(split_corpus.Data.values.tolist())
#     return vec


def bm25_vectorizer_exp(split_corpus):
    bm25 = BM25_Exp1()
    lines = []
    for items in split_corpus.Data.values.tolist():
        lines.append(items.split())
    vec = bm25.fit(lines)
    return vec


def bm25_vectorizer_cq(dataset):
    retriever = BM25Retriever(ngram_range=(1, 1), max_df=0.85, stop_words='english')
    temp = retriever.fit(dataset)
    return temp
