import streamlit as st
import pickle
import pandas as pd

with open ('pipeline.pkl','rb') as f :
    pipeline= pickle.load(f)
df=pd.read.csv(reddit_post.csv)
df= df.loc[df.selfnet.notna(),:]
st.dataframe(df.sample($))
test_docs =(doc for doc in df.sample)
prediction = pipeline.predict(test_docs)
st.write(prediction)
df =df.loc[: , ['self']]