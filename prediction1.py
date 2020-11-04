import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
from progressbar import ProgressBar
import lifelines
import os

train_csv = pd.read_csv('train.csv')



def prepare(name):
    index = []
    frag = glob.glob("{}/*".format(name))
    df = pd.DataFrame()

    pbar = ProgressBar()
    for i in pbar(frag):
        df = np.append(df, pd.read_csv(i).mean())

    df = pd.DataFrame(df.reshape(len(frag), 10))

    for i in range(0, len(frag)):
        index = np.append(index, os.path.splitext(frag[i].split('{}/'.format(name))[1])[0])

    df['segment_id'] = index
    df['segment_id'] = df['segment_id'].astype(int)
    if name == 'train':
        df = pd.merge(df, train_csv, on=['segment_id'], how='left')
    return (df)

train_means = prepare('train')
test_means = prepare('test')