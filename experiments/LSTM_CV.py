from numpy.random import seed
seed(5393)
from tensorflow import set_random_seed
set_random_seed(12011)
from random import seed
seed(12345)

# From https://www.kaggle.com/kentaroyoshioka47/cnn-with-batchnormalization-in-keras-94
## required for efficient GPU use
import tensorflow as tf
from keras.backend import tensorflow_backend
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

import os
import sys
import pandas as pd
import numpy as np

from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.layers.merge import concatenate
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers, losses
from keras.utils import to_categorical
from Attention import Attention

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

from tqdm import tqdm
from time import time
from generator import Generator
from glob import iglob as glob

BATCH_MAXIMUM_LEN = 1000

##################################################################
# Ordinal categorical loss (https://github.com/JHart96/keras_ordinal_categorical_crossentropy/blob/master/ordinal_categorical_crossentropy.py)
##################################################################
def loss(y_true, y_pred):
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)


def get_feats(label_f,
              batch_dir):  

    # Labels
    batch_labels = np.load(label_f)['labels']
    i = os.path.basename(label_f).split("_")[0]
    
    batch_features = []

    # Linguistic
    # if 'linguistic' in FEAT:
    #     with np.load(os.path.join(batch_dir, "{}_linguistic.npz".format(i))) as tmp:
    #         batch_ling = tmp['linguistic']
    #         batch_features.append(batch_ling)

    # # AFINN / VADER
    if 'afinn' in FEAT\
    or 'AFINN' in FEAT\
    or 'vader' in FEAT\
    or 'abusive' in FEAT\
    or 'hatebase' in FEAT:

        with np.load(os.path.join(batch_dir, "{}_lexicon.npz".format(i))) as tmp:
            
            if 'afinn' in FEAT or 'AFINN' in FEAT:
                batch_afinn = tmp['afinn']
                batch_afinn = np.concatenate(batch_afinn, axis = 0)\
                                   .reshape(-1, BATCH_MAXIMUM_LEN, feats_sizes['afinn']) #? x 1000 x 1
                
                batch_afinn = batch_afinn[:, -max_len:, :]

                batch_features.append(batch_afinn)
            
            if 'vader' in FEAT:
                batch_vader = tmp['vader']
                batch_vader = np.concatenate(batch_vader, axis = 0)\
                                .reshape(-1, BATCH_MAXIMUM_LEN, feats_sizes['vader'])
                
                batch_vader = batch_vader[:, -max_len:, :]

                batch_features.append(batch_vader)

            if 'abusive' in FEAT:
                batch_abusive = tmp['abusive_scores']
                batch_abusive = np.concatenate(batch_abusive, axis = 0)\
                                  .reshape(-1, BATCH_MAXIMUM_LEN, feats_sizes['abusive'])
                
                batch_abusive = batch_abusive[:, -max_len:, :]

                batch_features.append(batch_abusive)

            if 'hatebase' in FEAT:
                batch_hate = tmp['hatebase']
                batch_hate = np.concatenate(batch_hate, axis = 0)
                batch_hate = np.array(batch_hate)\
                               .reshape(-1, BATCH_MAXIMUM_LEN, feats_sizes['hatebase'])
                batch_hate = batch_hate[:, -max_len:, :]
                batch_features.append(batch_hate)

    # Empath_192
    if 'empath_192' in FEAT:
        with np.load(os.path.join(batch_dir, "{}_empath.npz".format(i))) as tmp:
            batch_empath = tmp['empath']
            batch_empath = np.concatenate(batch_empath, axis = 0)\
                             .reshape(-1, BATCH_MAXIMUM_LEN, 194)
            mask = np.ones(batch_empath.shape[2], dtype = bool)
            mask[[173, 192]] = False
            batch_empath = batch_empath[:, -max_len:, mask]
            batch_features.append(batch_empath)

    if 'empath_pos_neg' in FEAT\
    or 'empath_sent' in FEAT\
    or 'empath_2' in FEAT:
        with np.load(os.path.join(batch_dir, "{}_empath.npz".format(i))) as tmp:
            batch_empath = tmp['empath']
            batch_empath = np.concatenate(batch_empath, axis = 0)\
                             .reshape(-1, BATCH_MAXIMUM_LEN, 194)
            batch_empath = batch_empath[:, -max_len:, [173, 192]]
            batch_features.append(batch_empath)        

    # Ngrams
    if 'ngrams' in FEAT:
        with np.load(os.path.join(batch_dir, "{}_ngrams.npz".format(i))) as tmp:
            batch_ngrams = tmp['features'][:, -max_len:, :]
            batch_features.append(batch_ngrams) # ? x 1000 x VOCAB_SIZE

    # Word2Vec
    if 'w2v' in FEAT or 'word2vec' in FEAT:
        with np.load(os.path.join(batch_dir, "{}_word2vec.npz".format(i))) as tmp:
            batch_w2v = np.concatenate(tmp['features'], axis = 0)\
                          .reshape(-1, BATCH_MAXIMUM_LEN, feats_sizes['word2vec'])
            batch_w2v = batch_w2v[:, -max_len:, :]
            batch_features.append(batch_w2v)

    if len(batch_features) == 0:
        raise Exception("Empty features!!")

    batch_features = np.concatenate(batch_features, axis = 2)

    # Genre
    with np.load(os.path.join(batch_dir, "{}_meta.npz".format(i))) as tmp:
        batch_genre = tmp['genre'].reshape(-1, 1)
        batch_genre = to_categorical(batch_genre, num_classes = 12)
    
    return ([batch_features, batch_genre], batch_labels)

############################################################
#
############################################################
feats_sizes = {
    'word2vec': 300,
    'w2v': 300,
    'afinn': 1,
    'AFINN': 1,
    'vader': 1,
    'ngrams': 10000,
    'hatebase': 1018,
    'abusive': 1,
    'empath_192': 192,
    'empath_2': 2,
    # 'linguistic': 6 # Not implemented yet
}


FEAT = ['ngrams', 'afinn', 'vader', 'hatebase', 'empath_192', 'abusive', 'empath_2', 'w2v']
VOCAB_SIZE = sum(map(feats_sizes.__getitem__, FEAT))
max_len = 500
epochs = 30
batch_size = 16

############################################################
# Violence model
############################################################
def create_model():
    inp = Input(shape = (max_len, VOCAB_SIZE))
    x = Dropout(0.5)(inp)

    ############################################################
    # Encoder
    ############################################################
    x = LSTM(32, return_sequences=True)(x)
    x = Attention()(x)

    ############################################################
    # Genre model
    ############################################################
    genre_inp = Input(shape = (12,))
    x = concatenate([x, genre_inp])

    ############################################################
    # Classifier
    ############################################################
    x = Dense(3, activation="softmax", name = 'violence')(x)

    ############################################################
    # Model definition
    ############################################################
    model = Model(inputs=[inp, genre_inp], outputs=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def get_testdata(test_dir):
    files = list(glob(os.path.join(test_dir, "*_labels.npz")))
    labels, feats, genres = [], [], []

    for label_f in tqdm(files):
        (batch_features, batch_genre), batch_labels = get_feats(label_f, test_dir)

        feats.append(batch_features)
        genres.append(batch_genre)
        labels.append(batch_labels)

    # Concatenate lists of lists into tensor
    y_test = np.concatenate(labels, axis = 0).argmax(axis = 1)
    feats = np.concatenate(feats, axis = 0)
    genres = np.concatenate(genres, axis = 0)

    return y_test, feats, genres

############################################################
# TRAIN / EVAL
############################################################
def train_and_evaluate_model(model, data_train_gen, data_eval_folder, data_test_folder, data_eval_gen = None, epochs = 30, model_name = "best_model.hdf5"):
    
    # Load eval data
    y_eval, feats_eval, genre_eval = get_testdata(data_eval_folder)
    y_eval = to_categorical(y_eval, num_classes = 3)

    model.fit_generator(epochs = epochs,
                        generator = data_train_gen,
                        steps_per_epoch = len(data_train_gen),
                        validation_data = ([feats_eval, genre_eval], y_eval),
                        max_queue_size=30,
                        workers = 2,
                        use_multiprocessing = True,
                        callbacks = [
                                        EarlyStopping(monitor="val_loss", patience=5, verbose=1, mode="auto"),
                                        ModelCheckpoint(model_name, monitor='val_loss', verbose = 1, save_best_only = True)
                                    ])

    # Save memory
    feats_eval = None

    # Evaluate
    model = None #just to be absolutely sure
    model = create_model()
    model.load_weights(model_name)
    y_test, feats, genres = get_testdata(data_test_folder)

    # Model predictions
    y_pred = model.predict([feats, genres])
    y_pred = y_pred.argmax(axis = 1)

    return np.vstack([y_test, y_pred])

############################################################
#
############################################################
if __name__ == '__main__':

    folddir = sys.argv[1]
    outf = sys.argv[2]

    if len(sys.argv[3]) > 2:
        model_name = sys.argv[3]
    else:
        model_name = "models/best_model.hdf5"

    model = create_model()
    preds = train_and_evaluate_model(model,
                                     data_train_gen = Generator(f"{folddir}/train", get_feats),
                                     # data_eval_gen = Generator(f"{folddir}/eval", get_feats),
                                     data_eval_folder = f"{folddir}/eval",
                                     data_test_folder = f"{folddir}/test",
                                     model_name = model_name)
    y_test, y_pred = preds
    
    print(",".join(FEAT))
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average = 'macro')
    print(" ==== Results ==== ")
    print(f"{p}\t{r}\t{f1}")
    print("="*20)
    np.save(outf, preds)
    
