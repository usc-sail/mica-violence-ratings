import os
import threading
from random import Random
from glob import iglob as glob
from keras.utils import Sequence

class Generator(Sequence):
    def __init__(self, batch_dir, feat_func, shuffle = True, shuffler = Random(42)):
        self.batch_dir = batch_dir
        self.shuffler = shuffler
        self.shuffle = shuffle
        self.feat_func = feat_func
        self.files = list(glob(os.path.join(self.batch_dir, "*_labels.npz")))
        self.shuffler.shuffle(self.files)
        self.length = len(self.files)
        self.on_epoch_end()
        # print('generator initiated')

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffler.shuffle(self.files)

    def __getitem__(self, index):
        """Generates one batch of data"""
        # print(f'generator: {index}')
        label_f = self.files[index % self.length]
        return self.feat_func(label_f, self.batch_dir)

    def __len__(self):
        return self.length