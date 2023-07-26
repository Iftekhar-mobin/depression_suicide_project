import os

CORPUS_DIR = os.path.expanduser('~') + '/Documents/mental_health/suicide/'
CORPUS_NAME = 'Suicide_Detection.csv'
CORPUS_PATH = CORPUS_DIR + CORPUS_NAME
PROCESSED_CORPUS = 'Suicide_Detection_processed.csv'
PICKLED_CORPUS = 'Suicide_Detection_scatter.pkl'
SCATTERTEXT_FILE = 'Reddit_ScattertextRankDataJitter.html'
CORPUS_EQUAL_CLASS = 'Suicide_Detection_processed_equal_class.csv'
CSSR_DATASET = '500_Reddit_users_posts_labels.csv'
CSSR_DIR = CORPUS_DIR + 'CSSRS/'
BEST_ENTITIES = ['high school', 'mental health', 'best friend', 'feel like', 'really want', 'suicide thought',
                 'friend family']
CSSR_CAT = ['Indicator', 'Attempt','Behavior','Ideation']
CSSR_FILES = ['suicidal_indicator.csv', 'suicidal_attempt.csv', 'suicidal_behavior.csv', 'suicidal_ideation.csv']
