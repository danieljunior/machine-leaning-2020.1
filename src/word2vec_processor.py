import gensim

def train_w2v(corpus, filename, sg=0):
#     sg ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.
# In the CBOW model, the distributed representations of context (or surrounding words) 
# are combined to predict the word in the middle. 
# While in the Skip-gram model, the distributed representation of the input word is used 
# to predict the context.
# https://towardsdatascience.com/nlp-101-word2vec-skip-gram-and-cbow-93512ee24314
    model = gensim.models.Word2Vec(sentences=corpus, min_count=50, size=100, workers=4, sg=sg)
    model.save(filename)
    return model

def load_w2v(path='model.bin'):
    model = gensim.models.Word2Vec.load(path)
    return model

def load_pretrained_w2v(path):
    model = gensim.models.KeyedVectors.load_word2vec_format(path)
    return model