import pandas as pd
import scattertext as st
import seaborn as sns
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews import opts, dim
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sns.set_style("darkgrid")


def create_bar_plot(dataframe, xl, yl, title):
    dataframe.plot(kind="bar")
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(title)
    plt.show()


def create_hor_bar_plot(dataframe, title):
    dataframe.plot(kind="barh")
    plt.legend(['Frequency'])
    plt.xlabel("Terms Frequency")
    plt.ylabel("Terms")
    plt.title(title)
    plt.show()


def create_scatter_plot(scatter_data, corpus, save_path):
    # CREATING A SCATTERTEXT PLOT
    html = st.produce_scattertext_explorer(corpus,
                                           category='suicide',
                                           category_name='Suicide',
                                           not_category_name='Depression',
                                           width_in_pixels=1000,
                                           jitter=0.1,
                                           minimum_term_frequency=5,
                                           transform=st.Scalers.percentile,
                                           metadata=scatter_data['class']
                                           )
    open(save_path, 'wb').write(html.encode('utf-8'))


def create_chord_chart(node_data, encoded_df):
    hv.extension('bokeh')
    hv.output(size=200)
    # add node labels
    nodes = hv.Dataset(pd.DataFrame(node_data['nodes']), 'index')
    # create chord object
    chord = hv.Chord((encoded_df, nodes)).select(value=(5, None))
    # customization of chart
    chord.opts(
        opts.Chord(cmap='Category20', edge_cmap='Category20',
                   edge_color=dim('source').str(),
                   labels='nodes', node_color=dim('index').str()))


def prepare_risk_count():
    df = pd.DataFrame([['Depression', 2, 74, 112929, 2971], ['Suicide', 7, 146, 115014, 853]],
                      columns=['Category', 'Attempt', 'Behavior', 'Ideation', 'Indicator'])

    # plot grouped bar chart
    df.plot(x='Category',
            kind='bar',
            log=True,
            rot=0,
            stacked=False)
    plt.xlabel('Category')
    plt.ylabel('Number of samples')
    plt.title('Suicidal Risk Estimation')
    plt.show()

    vec_1 = np.array([[2, 74, 112929, 2971]])
    vec_2 = np.array([[7, 146, 115014, 853]])
    return cosine_similarity(vec_1, vec_2)
