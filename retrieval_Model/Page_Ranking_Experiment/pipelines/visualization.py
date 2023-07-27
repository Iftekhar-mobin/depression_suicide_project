import matplotlib.pylab as plt
import pandas as pd


def visualizer(rank):
    col = []
    for ids, score in rank:
        col.append(ids)

    df = pd.DataFrame(rank, columns=['PageID', 'Score'], index=col)
    df['Score'].plot.bar(x='PageID', y='Score')
    plt.show()
