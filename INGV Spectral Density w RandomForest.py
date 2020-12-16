import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from scipy import signal
import seaborn as sns
import glob

DATAPATH='/media/jeonghn/C684587984586E45'
train_labels=pd.read_csv(DATAPATH+'/data/train.csv')
train_labels.head()

print(f"Total segment files: {len(train_labels['segment_id'])}")
train_labels.dtypes

df_example=pd.read_csv(DATAPATH+'/data/train/1003154738.csv')
df_example.head()
type(df_example.columns)
data_columns=list(df_example.columns)
df_example.shape
df_example.describe()

fig,axs=plt.subplots(nrows=5, ncols=2)
fig.set_size_inches(20,10)
fig.subplots_adjust(hspace=0.5)

for col, ax in zip(data_columns, axs.flatten()):
    ax.plot(range(len(df_example[col])), df_example[col])
    ax.set_title(col)

plt.show()
