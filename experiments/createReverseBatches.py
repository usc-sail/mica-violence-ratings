from numpy.random import seed
seed(5393)
from tensorflow import set_random_seed
set_random_seed(12011)

import os

import numpy as np
import pandas as pd 
from scipy import sparse

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold

from joblib import Parallel, delayed
from tqdm import tqdm

import logging
logging.basicConfig(level = logging.INFO)

EMBED_DIM = 300
VOCAB_SIZE = 5000
max_len = 1000
batch_size = 16
n_folds = 5
fold_dir = "/data/victor/violence-workshop/batches/reversefolds"
data_pkl = "../../data/dataframe_with_scores_withdoc2vec.pkl"

def pad_csr(a, newshape):
    """ Pads csr_matrix with zeros. Modifies a inplace. """
    n, m = a.shape
    a._shape = newshape
    a.indptr = np.pad(a.indptr, (0, newshape[0] - n), 'edge')

def filter_nans(seq):
    """ Filters out floats (np.nan) from list """
    return np.array([x for x in seq if not isinstance(x, float)])

def pad_or_trim(seq, max_len=1000):
    """ Pads or trims seq to have max_len rows """
    n, m = seq.shape
   
    if n > max_len:
        seq = seq[-max_len:, :]
    elif n < max_len:
        if sparse.issparse(seq):
            pad_csr(seq, (max_len, m))
        else:
            seq = np.r_[seq, np.zeros((max_len - n, m))]
    return seq

def process_ngrams(batch_features, ngram_features):
    """ Transform batch_features into tensor of dims:
     (n, max_len, #features) where n is len(batch_features)"""
    n = batch_features.shape[0]

    batch_features = batch_features.apply(ngram_features.transform)\
                                   .apply(pad_or_trim)

    batch_features = sparse.vstack(batch_features)

    batch_features = batch_features.toarray()\
                                   .reshape(n, max_len, -1)

    return batch_features

def process_scores(X):
    """ Transforms X into tensor of dims:
    (n, max_len, #features) where n is len(X).
    
    This is a special case of process for lists of scores"""
    batch_scores =  X.apply(np.array)\
                     .apply(lambda x: x.reshape(-1, 1))\
                     .apply(pad_or_trim)

    batch_scores = np.concatenate(batch_scores.values, axis = 0)\
                     .reshape(-1, max_len, 1)

    return batch_scores


############################################################
# Load Data
############################################################
data = pd.read_pickle(data_pkl)

# Encode genre
lb_genre = LabelEncoder()
data['genre'] = lb_genre.fit_transform(data['genre'])


############################################################
# 3 to 5 chars w/ spaces
# unigrams + bigrams
############################################################
# This defines the analyzer to be used with Countvectorizer
def char_ngram_tokenizer(text, ngram_range):
    def aux(text, ngram_size):
        for i in range(len(text) - ngram_size):
            yield text[i : i + ngram_size]

    for n in range(*ngram_range):
        for ngram in aux(text, n):
                yield ngram
                       
ngram_features = FeatureUnion([
    ("char_ngrams", CountVectorizer(analyzer = lambda text: char_ngram_tokenizer(text, ngram_range=(3, 6)),
                                    max_features = VOCAB_SIZE)),
    ("token_ngrams", CountVectorizer(ngram_range=(1, 2),
                                     max_features=VOCAB_SIZE))
    ])

tfidf_ = TfidfVectorizer(ngram_range=(1, 2), max_features=VOCAB_SIZE)

############################################################
# Batch generation
############################################################
def process(X, Y, i, ngram_features, batch_dir, tfidf_transformer = None):
    # Features
    ## ngrams
    #logging.info("ngrams")
    #batch_ngrams = process_ngrams(X['sentences'].iloc[i : i + batch_size], ngram_features)
    #np.savez(os.path.join(batch_dir, "{}_ngrams".format(i)),
    #    features = batch_ngrams)
    #batch_ngrams = None
    
    ## tfidf
    #logging.info("tfidf")
    #batch_tfidf = process_ngrams(X['sentences'].iloc[i : i + batch_size], tfidf_transformer)
    #np.savez(os.path.join(batch_dir, "{}_tfidf".format(i)),
    #    features = batch_tfidf)
    #batch_tfidf = None

    # ## Word2vec
    #logging.info("word2vec")
    #batch_word2vec = X['word2vec_sent_mean_vec'].iloc[i : i + batch_size]\
    #                                            .apply(filter_nans)\
    #                                            .apply(pad_or_trim)
    #np.savez(os.path.join(batch_dir, "{}_word2vec".format(i)),
    #    features = batch_word2vec)
    #batch_word2vec = None

    # paragraph2vec
    logging.info("paragraph2vec")
    batch_paragraph2vec = X['doc2vec_vectors'].iloc[i : i + batch_size]\
					      .apply(filter_nans)\
					      .apply(pad_or_trim)
    np.savez(os.path.join(batch_dir, "{}_doc2vec".format(i)),
             features = batch_paragraph2vec)
    batch_paragraph2vec = None

    # ## Lexicons
    #logging.info("Empath")
    #batch_empath = X['empath_sentence'].iloc[i : i + batch_size]\
    #                                    .apply(np.array)\
    #                                    .apply(pad_or_trim)
    #np.savez(os.path.join(batch_dir, "{}_empath".format(i)),
    #         empath = batch_empath)
    #logging.info("Lexicons")
    #batch_lexicon = process_scores(X['abusive_scores'].iloc[i : i + batch_size])
    #batch_vader = process_scores(X['vader_scores'].iloc[i : i + batch_size])
    #batch_afinn = process_scores(X['afinn_scores'].iloc[i : i + batch_size])
    #batch_hatebase = X['hatebase_sentence'].iloc[i : i + batch_size].apply(pad_or_trim)
    #np.savez(os.path.join(batch_dir, "{}_lexicon".format(i)),
    #    abusive_scores = batch_lexicon,
    #    vader = batch_vader,
    #    afinn = batch_afinn,
    #    hatebase = batch_hatebase)

   # batch_lexicon = None
    #batch_vader = None
    #batch_afinn = None
    #batch_hatebase = None

    ## Save labels
    #logging.info("Labels")
    #batch_labels = Y[i : i + batch_size]
    #np.savez(os.path.join(batch_dir, "{}_labels".format(i)),
    #    labels = batch_labels)


    ## Save metadata
    #logging.info("Metadata")
    #batch_genre = X['genre'][i : i + batch_size]
    #np.savez(os.path.join(batch_dir, "{}_meta".format(i)),
    #    genre = batch_genre)

    logging.info("Done for {}".format(i))


skf = StratifiedKFold(n_splits = n_folds, random_state = 42)
lb = LabelBinarizer()
Y = lb.fit_transform(data['violence_rating'])

for k, (train, test) in enumerate(skf.split(data.violence_rating, data.violence_rating)):
    
    train_dir = os.path.join(fold_dir, str(k), "train")
    test_dir = os.path.join(fold_dir, str(k), "test")
    eval_dir = os.path.join(fold_dir, str(k), "eval")

    for t in [train_dir, test_dir, eval_dir]:
        os.makedirs(t, exist_ok = True)

    X_train, X_test = data.iloc[train], data.iloc[test]
    Y_train, Y_test = Y[train], Y[test]
    X_train, X_eval, Y_train, Y_eval = train_test_split(X_train, Y_train, test_size = 64, random_state = 666)

    # Fit vocab
    ngram_features.fit(data.iloc[train]['text'], Y_train)
    tfidf_.fit(data.iloc[train]['text'], Y_train)

    # Create batches
    for i in tqdm(range(0, X_train.shape[0], batch_size)):
        process(X_train, Y_train, i, ngram_features = ngram_features, batch_dir = train_dir, tfidf_transformer = tfidf_)

    for i in tqdm(range(0, X_eval.shape[0], batch_size)):
        process(X_eval, Y_eval, i, ngram_features = ngram_features, batch_dir = eval_dir, tfidf_transformer = tfidf_)

    for i in tqdm(range(0, X_test.shape[0], batch_size)):
        process(X_test, Y_test, i, ngram_features = ngram_features, batch_dir = test_dir, tfidf_transformer = tfidf_)
