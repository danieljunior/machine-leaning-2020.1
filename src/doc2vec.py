import numpy as np
from tcu_corpus import TCUCorpus
# source: https://github.com/v1shwa/document-similarity/blob/master/DocSim.py
class Doc2Vec:
    
    def __init__(self, w2v_model):
        self.w2v_model = w2v_model
        
    def transform(self, docs):
        return np.array([self.vectorize(doc) for doc in docs])

    def vectorize(self, doc: str) -> np.ndarray:
        """
        Identify the vector values for each word in the given document
        :param doc:
        :return:
        """
        
        words = TCUCorpus.preprocess(doc)
        word_vecs = []
        for word in words:
            try:
                vec = self.w2v_model.wv[word]
                word_vecs.append(vec)
            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                pass

        # Assuming that document vector is the mean of all the word vectors
        # PS: There are other & better ways to do it.
        vector = np.mean(word_vecs, axis=0)
        return vector