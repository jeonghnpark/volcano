import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

import glob
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import math
from sklearn.metrics import mean_squared_error as mse

import optuna
from optuna.samplers import TPESampler
pd.set_option('display.max_columns', None)

from sklearn.feature_selection import RFE

train=pd.read_csv('/media/jeonghn/C684587984586E45/data/train.csv')
sample_submission=pd.read_csv('/media/jeonghn/C684587984586E45/data/sample_submission.csv')
sample_submission.shape


fig=px.histogram(train, x='time_to_eruption', nbins=200)
fig.show()
fig2=px.line(train, y='time_to_eruption',title='time to eruption')
fig2.show()

train_frags=glob.glob('/media/jeonghn/C684587984586E45/data/train/*')
test_frags=glob.glob('/media/jeonghn/C684587984586E45/data/test/*')
len(test_frags)
len(train_frags)

check=pd.read_csv(train_frags[0])
len(check)

#check totally missed sensor for each volcano
null_sensors=list()
null_sensor_groups=list()

for i,frag in enumerate(train_frags[:100]):
    # print(i, frag)
    df_frag=pd.read_csv(frag)
    # null_sensor_group=list()
    for sensor in df_frag:
        # print(sensor)
        if df_frag[sensor].isnull().all()==True:
            null_sensors.append(sensor)
            # null_sensor_group.append(sensor)
    # null_sensor_groups.append(null_sensor_group)
            print(sensor+' of '+frag +' is totally null')

null_sensors_dic={}
for ns in null_sensors:
    if ns in null_sensors_dic:
        null_sensors_dic[ns]+=1
    if ns not in null_sensors_dic:
        null_sensors_dic[ns]=1

#visualize dictionary
null_sensors_df=pd.DataFrame(null_sensors_dic.items(), columns=['Sensors', 'number of missed sensors'])
fig3=px.bar(null_sensors_df, x="Sensors", y='number of missed sensors')
fig3.show()


null_sensor_groups=list()
for i,frag in enumerate(train_frags[:]):
    df_frag=pd.read_csv(frag)
    null_sensor_group=list()
    for sensor in df_frag:
        if df_frag[sensor].isnull().all()==True:
            null_sensor_group.append(sensor)
    if len(null_sensor_group) > 0:
        null_sensor_groups.append(null_sensor_group)

null_sensor_groups_dic={}
for nsg in null_sensor_groups:
    print(nsg)
    if str(nsg) in null_sensor_groups_dic:
        null_sensor_groups_dic[str(nsg)]+=1
    if str(nsg) not in null_sensor_groups_dic:
        null_sensor_groups_dic[str(nsg)]=1

null_sensor_groups_df=pd.DataFrame(null_sensor_groups_dic.items(), columns=['sensor group', 'number of missed sensor goup'])
null_sensor_groups_df=null_sensor_groups_df.sort_values('number of missed sensor goup')
fig4=px.bar(null_sensor_groups_df,y='sensor group', x='number of missed sensor goup',orientation='h')
fig4.show()



check=pd.read_csv('/media/jeonghn/C684587984586E45/data/train/1002275321.csv')
check