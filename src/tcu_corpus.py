import nltk
import re
from gensim import utils
from tqdm import tqdm
import pandas as pd

nltk.download('stopwords')


class TCUCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __init__(self, filename, sample=None):
        self.filename = filename
        self.data = pd.read_csv(self.filename)
        if sample:
            self.data = (self.data.groupby(sample['class'])
                                    .apply(pd.DataFrame.sample, frac=sample['frac'])
                                    .reset_index(drop=True))

    @staticmethod
    def clean_text(text):
        # source: https://medium.com/ml2vec/using-word2vec-to-analyze-reddit-comments-28945d8cee57

        # Normalize tabs and remove newlines
        no_tabs = str(text).replace('\t', ' ').replace('\n', '')

        # Remove all characters except A-Z
        alphas_only = re.sub("[^a-zA-Z]", " ", no_tabs)

        # Normalize spaces to 1
        multi_spaces = re.sub(" +", " ", alphas_only)

        # Strip trailing and leading spaces
        no_spaces = multi_spaces.strip()

        # Remove stopwords
        stopwords = nltk.corpus.stopwords.words('portuguese')
        clean_text = [w for w in no_spaces.split() if not w in stopwords]

        return ' '.join(clean_text)

    @staticmethod
    def preprocess(text):
        return utils.simple_preprocess(TCUCorpus.clean_text(text))

    def __iter__(self):
        for index, row in tqdm(self.data.iterrows()):
            yield TCUCorpus.preprocess(row.acordao)
