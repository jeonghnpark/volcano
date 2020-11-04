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
fig = px.histogram(train, x="time_to_eruption", width=800, height=500,
                   nbins=100, title="Time to eruption distribution")

# fig.show()

fig = px.line(train, y='time_to_eruption', width=800, height=500,
              title="time to eruption distribution")
# fig.show()

train_frags = glob.glob("train/*")
len(train_frags)
train_frags[0]
'.csv' in train_frags
check = pd.read_csv(train_frags[0])
check.head()

# check the number of observation
sensors = set()
observation = set()
nan_columns = list()
missed_groups = list()

a_frag = pd.read_csv(train_frags[0])

for_df = list()

for item in train_frags[:50]:
    name = int(item.split('.')[-2].split('\\')[-1])
    at_least_one_missed = 0
    frag = pd.read_csv(item)
    missed_group = list()
    missed_percents = list()
    for col in frag.columns:
        missed_percents.append(frag[col].isnull().sum() / len(frag))
        if pd.isnull(frag[col]).all() == True:
            print(name, col)
            at_least_one_missed = 1
            nan_columns.append(col)  # type(col)==<class 'pandas.core.series.Series'>
            missed_group.append(col)
    if len(missed_group) > 0:
        missed_groups.append(missed_group)
    sensors.add(len(frag.columns))
    observation.add(len(frag))
    for_df.append([name, at_least_one_missed] + missed_percents)

print('Unique number of sensors: ', sensors)
print('Unique number of obs', observation)
print('number of totaly missed sensors', len(nan_columns))

absent_sensors=dict()

