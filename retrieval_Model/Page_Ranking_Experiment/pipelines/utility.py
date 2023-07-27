from itertools import groupby


def get_unique_2Dlist(list_2D):
    lists = sorted(list_2D) # Necessary step to use groupby
    grouped_list = groupby(lists, lambda x: x[0])
    grouped_list = [(x[0], [k[1] for k in list(x[1])]) for x in grouped_list]
    
    sorted_list = []
    for ids, sentences in grouped_list:
        sorted_list.append([ids, list(set(sentences))])
    return sorted_list
