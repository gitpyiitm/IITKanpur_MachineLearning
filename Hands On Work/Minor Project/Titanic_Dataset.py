# -*- coding: utf-8 -*-

#collect dataset
#prepare dataset

import pandas as pd

df = pd.read_csv("dataset/train.csv")

df.head(5) #to display top 5 records

df.shape #know the shape of your dataset

df.columns # know the name of col keys

df.Survived.value_counts()

df[df.Sex=='male'].Survived.value_counts().sum()

df[['Sex','Survived']].groupby("Sex",as_index=False).mean()
#as_index=false displays indexes

for i in df.groupby("Pclass"):
    print(i)
    
df.groupby("Pclass").mean()

df1.isnull().sum()

df1.fillna({"Age":d1.Age.mean()})
df3.select_dtypes(exclude=["object"])
