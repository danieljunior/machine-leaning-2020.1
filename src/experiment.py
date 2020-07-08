from tcu_corpus import TCUCorpus
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import word2vec_processor
from doc2vec import Doc2Vec
import logging
import pickle
from joblib import Parallel, delayed

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%d/%m/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def generate_tfidf_vectorizer(corpus):
    try:
        tfidf = pickle.load(open("tfidf.pickle", "rb"))
    except:
        tfidf = TfidfVectorizer()
        tfidf.fit([TCUCorpus.clean_text(doc) for doc in corpus.data.acordao])
        pickle.dump(tfidf, open("tfidf.pickle", "wb"))

    return tfidf

def step_validation(train_index, test_index, train_encoded_data, ci, clf, 
                    clfs_map, embedders_map):
    step_results = []
    for ei, X_enc_train in enumerate(train_encoded_data):
        X_, X_validation = X_enc_train[train_index], X_enc_train[test_index]
        y_, y_validation = y_train[train_index], y_train[test_index]

        startTime = time.perf_counter()

        clf.fit(X_, y_)
        y_pred = clf.predict(X_validation)

        elapsed_time = time.perf_counter() - startTime

        p, r, f, s = precision_recall_fscore_support(
            y_validation, y_pred, average='micro')

        result = {"CLASSIFIER": clfs_map[ci],
                    "EMBEDDER": embedders_map[ei],
                    "PRECISION": p,
                    "RECALL": r,
                    "F1SCORE": f,
                    "TIME": elapsed_time}
        step_results.append(result)
    return step_results

flatten = lambda l: [item for sublist in l for item in sublist]


logger.info("Lendo dados...")
corpus = TCUCorpus('/app/datasets/acordaos_relator_5k.csv', 
                   sample={'frac': 0.1, 'class': 'relator'})

logger.info("Carregando modelo word2Vec treinado na base...")
try:
    model_trained = word2vec_processor.load_w2v('/app/model.bin')
except:
    model_trained = word2vec_processor.train_w2v(corpus, '/app/model.bin')
    
logger.info("Carregando modelo word2Vec pré-treinado...")
# http://www.nilc.icmc.usp.br/embeddings - CBOW 100
model_pretrained = word2vec_processor.load_pretrained_w2v('/app/cbow_s100.txt')

logger.info("Gerando vetorizador TFIDF...")
tfidf_vectorizer = generate_tfidf_vectorizer(corpus)

logger.info("Codificando y...")
X = np.array(corpus.data.acordao)
l_enc = preprocessing.LabelEncoder()
y = l_enc.fit_transform(np.array(corpus.data.relator.tolist()))

logger.info("Dividindo dados para treinamento e teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

embedders = [tfidf_vectorizer, Doc2Vec(
    model_pretrained), Doc2Vec(model_trained)]

embedders_map = {0: 'TFIDF', 1: 'PRETRAINED', 2: 'TRAINED'}

logger.info("Pré-codificando dados em cada modelo de vetorização...")
try:
    train_encoded_data = pickle.load(open("train_enc.pickle", "rb"))
    test_encoded_data = pickle.load(open("test_enc.pickle", "rb"))
except:
    train_encoded_data = []
    test_encoded_data = []
    for embedder in tqdm(embedders):
        X_train_enc = embedder.transform(X_train)
        train_encoded_data.append(X_train_enc)
        X_test_enc = embedder.transform(X_test)
        test_encoded_data.append(X_test_enc)
    pickle.dump(train_encoded_data, open("train_enc.pickle", "wb"))
    pickle.dump(test_encoded_data, open("test_enc.pickle", "wb"))
    
clfs = [KNeighborsClassifier(n_neighbors=7),
        SVC(gamma='auto', random_state=42),
        MLPClassifier(random_state=42, max_iter=300)]
clfs_map = {0: 'KNN', 1: 'SVM', 2: 'MLP'}

logger.info("Realizando K-Fold Validation...")

validation_results_rows = []
kf = KFold(n_splits=10)
for train_index, test_index in tqdm(kf.split(X_train)):
    processed_list = Parallel(n_jobs=3)(delayed(step_validation)(train_index, test_index, train_encoded_data, ci, clf, clfs_map, embedders_map) 
                                        for ci, clf in enumerate(clfs))
    validation_results_rows.append(flatten(processed_list))
            
logger.info("Salvando dados de validação...")
validation_results = pd.DataFrame(flatten(validation_results_rows))
validation_results.to_csv('validation_results.csv')

logger.info("Realizando treinamento e teste...")
test_results_rows = []
for ci, clf in enumerate(clfs):
    for ei, X_enc_test in enumerate(test_encoded_data):
        X_enc_train = train_encoded_data[ei]
        startTime = time.perf_counter()

        clf.fit(X_enc_train, y_train)
        y_pred = clf.predict(X_enc_test)

        elapsed_time = time.perf_counter() - startTime

        p, r, f, s = precision_recall_fscore_support(
            y_test, y_pred, average='micro')

        result = {"CLASSIFIER": clfs_map[ci],
                  "EMBEDDER": embedders_map[ei],
                  "PRECISION": p,
                  "RECALL": r,
                  "F1SCORE": f,
                  "TIME": elapsed_time}
        test_results_rows.append(result)
        logger.info("Resultado: %s" % str(result))

logger.info("Salvando dados de teste...")
test_results = pd.DataFrame(test_results_rows)
test_results.to_csv('test_results.csv')

logger.info("Finalizado!")
