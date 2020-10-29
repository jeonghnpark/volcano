import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import glob

train = pd.read_csv('train.csv')
sample_submission = pd.read_csv('sample_submission.csv')
train
hist, bin_edges = np.histogram(train['time_to_eruption'], bins=100)
fig = plt.hist(hist, bins='auto')

# another way of draw histogram using "plotly" package
# plotly get input variable DataFrame
fig=px.histogram(train, x="time_to_eruption", width=800, height=500,
                 nbins=100, title="Time to eruption distribution")

fig.show()

fig=px.line(train, y='time_to_eruption', width=800, height=500,
            title="time to eruption distribution")
fig.show()

train_frags=glob.glob("train/*")
len(train_frags)
train_frags[0]
check=pd.read_csv(train_frags[0])
check.head()

#check the number of observation
sensors=set()
observation=set()
nan_columns=list()
missing_groups=list()

for_df=list()

for item in train_frags:
    pass

train_frags[0].split('.')[-2].split('\\')[-1]

first_item=pd.read_csv(train_frags[0])
type(first_item)


