import pandas as pd
import numpy as np
import matplotlib as mb
import dplyr
df = pd.read_csv('C:\_bike_share.csv')

df=df.dropna(subset=['start_station_id'])
df.head()

pd.value_counts(df.member_gender)

df['member_gender'].fillna("Other")

df = df.loc[:,('start_station_id','end_station_id')]

df

df.isna().sum()

df=df.loc[df.start_station_id.notna(),:]

df.shape

df1 = df[df.start_station_id == 15]

df1[df1.end_station_id ==30]