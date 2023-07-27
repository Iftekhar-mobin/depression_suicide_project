from .retriever_sklearn import BM25Retriever

retriever = BM25Retriever(ngram_range=(1, 1), max_df=0.85, stop_words='english')
