import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
from progressbar import ProgressBar
import lifelines
import os
import plotly.express as px

train = pd.read_csv('train.csv')

fig = px.histogram(train, x='time_to_eruption', nbins=200)
fig.show()
fig = px.line(train, y='time_to_eruption')
fig.show()

train.describe()

print("median", train['time_to_eruption'].median())
print("skew", train['time_to_eruption'].skew())
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission.head()
train.head()
train.info()
sample_submission.info()
(sample_submission['time_to_eruption'] == 0).all() == True
fig = px.line(sample_submission, y='time_to_eruption')
fig.show()

train_frag = glob.glob("train/*")  # return LIST of string of file path
test_frag = glob.glob("test/*")

check = pd.read_csv(train_frag[0])

# 1136037770 in train['segment_id']  # why error?

for e in train['segment_id']:
    print(type(e), e)

df = pd.DataFrame(np.array([1, 2, 3, 4]))
df.info()

1 in df[0]
df.columns = ['num']
df['num']

1 in df['num']

sensors = set()
observations = set()
nan_columns = list()
missed_groups = list()
for_df = list()

frag = pd.read_csv(train_frag[0])
frag.info()
len(frag)

i=0
for item in train_frag:
    i+=1
    name = int(item.split('.')[-2].split("\\")[-1])
    print(name, f" {i}/{len(train_frag)}")
    frag = pd.read_csv(item)
    at_least_one_missed = 0
    missed_group = list()
    missed_percents = list()
    for sensor_num in frag.columns:
        # print(sensor_num)
        missed_percents.append(frag[sensor_num].isnull().sum() / len(frag))
        # print(missed_percents)
        if pd.isnull(frag[sensor_num]).all()==True:
            at_least_one_missed=1
            nan_columns.append(sensor_num) #saving ['sensor_1', 'sensor_1', 'sensor_2'...]
            missed_group.append(sensor_num)

    if len(missed_group) > 0:
        missed_groups.append(missed_group)
    sensors.add(len(frag.columns))
    observations.add(len(frag))
    for_df.append([name, at_least_one_missed]+missed_percents)

print('unique number of sensors:', sensors)
print('unique number of obs', observations)

print('number of totaly missed sensors: ', len(nan_columns))

absent_sensor=dict()
for item in nan_columns:
    if item in absent_sensor:
        absent_sensor[item] +=1
    else:
        absent_sensor[item]=0 #missing 1 ??=> absent_sensor[item]=1

absent_df=pd.DataFrame(absent_sensor.items(), columns=['sensor','missed sensors'])

fig=px.histogram(absent_df,x='sensor', y='missed sensors', width=800, height=500, title='number of missed sensors in trainging dataset')
fig.show()

#missed combination of the sensors
absent_groups=dict()
for item in missed_groups:
    print(type(str(item)))
    print(str(item))
    if str(item) in absent_groups:
        absent_groups[str(item)]+=1
    else:
        absent_groups[str(item)]=0

absent_df=pd.DataFrame(absent_groups.items(), columns=['Group','Missed Number' ])
absent_df=absent_df.sort_values("Missed Number")
fig=px.bar(absent_df, y='Group', x="Missed Number", orientation='h',
           width=800, height=500, title="number of missed sensor group")
fig.show()

#change list to df
for_df=pd.DataFrame(for_df, columns=['segment_id', 'has_missed_sensors', 'missed_sensor1', 'missed_sensor2', 'missed_sensor3','missed_sensor4','missed_sensor5','missed_sensor6','missed_sensor7','missed_sensor8','missed_sensor9','missed_sensor10'])

train

train=pd.merge(train, for_df)
train.info()


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


# train_means = prepare('train')
# test_means = prepare('test')
