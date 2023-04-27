import streamlit as st
import pandas as pd
from google.cloud import firestore
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from google.cloud import storage
import tempfile
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import spacy
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import cross_validate

# your google credentials json file 
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "allie01233-330cd3847b13.json"

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

st.title("Sentiment Analyzer")

home, model, team = st.tabs(["Home", "Model", "Team"])

with home:
    # getting data from your firestore database - reddit collection
    db = firestore.Client()
    query = db.collection(u'rr').order_by(u'created', direction=firestore.Query.DESCENDING)
    posts = list(query.stream())
    docs_dict = list(map(lambda x: x.to_dict(), posts))
    df = pd.DataFrame(docs_dict)
    print(df.head())
    created_end = datetime.fromtimestamp(df.iloc[:1,:].created.values[0])
    created_start = datetime.fromtimestamp(df.iloc[-1:,:].created.values[0])

    #filter on sidebar
    date_start = st.sidebar.date_input("From", value=created_start, min_value=created_start, max_value=created_end)
    date_end = st.sidebar.date_input("To", value=created_end, min_value=created_start, max_value=created_end)
    posts_length_range = st.sidebar.slider("Posts Length", min_value=1, max_value=9999, value=[1, 9999])

    #processing data filter
    date_start_str = date_start.strftime('%Y-%m-%d')
    date_end_str = date_end.strftime('%Y-%m-%d')
    df['date'] = df['created'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
    df = df.loc[(df.date >= date_start_str) & (df.date <= date_end_str), :]
    
    #processing the posts length filter
    df['length'] = df['selftext'].apply(lambda x: len(x))
    df = df.loc[(df.length >= posts_length_range[0]) & (df.length <= posts_length_range[1]), :]

    #processing the data class
    Type=st.sidebar.radio("Type of posts", ['Positive', 'Negative', 'All'], key="1")
    if Type=="Positive":
        df=df[df.Liked==1]
    elif Type=="Negative":
        df=df[df.Liked==0]
    else :df=df
    
    chart1, desc1 = st.columns([2,1])
    with chart1: 
        fig1 = px.histogram(df, x='date', color='Liked', color_discrete_map={0: "blue", 1: "tomato"}, barmode="group")
        st.plotly_chart(fig1)
        st.caption("Sentiment on Subreddit r/music")
    with desc1:
        st.subheader("Music Review Sentiment on Reddit")
        st.write("Our sentiment model we have been able predit heartfelt \
                 sentiment of music reviews on reddit. The sentiment of user \
                 posts on subreddit 'r/music'. The graph shows the \
                 number user posts (positive and negative) for each \
                 day for the selected date range.")

    "---"
    #line chart for posts number per day
    chart2, desc2 = st.columns([2,1])
    with chart2:
        df_time=df.groupby('date').count()['length']
        chart2=st.line_chart(df_time)
        mean_posts=round(df.groupby('date').count()['length'].mean(),0)
        st.caption("Daily posting number")
        #st.markdown("The average number of updating reviews in the dataset is **{:.0f} words**.".format(mean_posts))
    with desc2:
        st.subheader("Number of daily updated reviews")
        st.write(f"The graph shows daily updating number of posts\
                 The average number of updating reviews in the dataset is {mean_posts}")
    "---"
    
    #bin chart for peak hours
    chart3, desc3 = st.columns([2,1])
    with chart3:
        df['hours']=df['created'].apply(lambda x: datetime.fromtimestamp(x).hour)
        df_hour=df.groupby('hours').count()['length']
        print(df_hour)
        chart2=st.bar_chart(df_hour)
        st.caption("Time period analysis of posting")
    with desc3:
        st.subheader("People like night talking")
        st.write("This graph shows number of posts of each hour.")
    "---"

    # World Cloud
    df_pos = df.loc[df.Liked == 1, ["selftext"]]
    df_neg = df.loc[df.Liked == 0, ["selftext"]]
    
    col1, col2 = st.columns(2)
    if Type=='Positive':
        with col1:
            st.subheader("Positive Posts")
            st.image(WordCloud().generate("/n".join(list(df_pos.selftext))).to_image())
        with col2:
            st.empty()
    elif Type=='Negative':
        with col1:
            st.empty()
        with col2:
            st.subheader("Negative Posts")
            st.image(WordCloud().generate("/n".join(list(df_neg.selftext))).to_image())
    else: 
        with col1:
            st.subheader("Positive Posts")
            st.image(WordCloud().generate("/n".join(list(df_pos.selftext))).to_image())
        with col2:
            st.subheader("Negative Posts")
            st.image(WordCloud().generate("/n".join(list(df_neg.selftext))).to_image())
    "---"

    ##most common words in whole data
    st.subheader('High Tf-idf words')
    st.image(WordCloud().generate("/n".join(['album', 'song', 'sound', 'like', 'record', 'good', 'band', 'music',
       'pop', 'time', 'feel', 'work', 'track', 'rock', 'new', 'little', 'come',
       'way', 'moment', 'year', 'listen', 'find', 'debut', 'love', 'great',
       'oct', 'thing', 'apr', 'nov', 'lack', 'hard', 'voice', 'mar', 'jun',
       'result', 'sound like', 'long', 'bad', 'lyric', 'vocal', 'sep', 'fan',
       'production', 'guitar', 'bit', 'far', 'leave', 'end', 'jul', 'musical'])).to_image())
    
    ##show the length of posts histogram
    chart4, desc4 = st.columns([2,1])
    with chart4:
        df['post_length']=df['selftext'].apply(lambda x:len(x.split()))
        fig4 = px.histogram(df, y='post_length', nbins=10)
        chart4=st.plotly_chart(fig4)
        st.caption("Length analysis of posting")
    with desc4:
        mean_length=round(df.post_length.mean(),0)
        st.subheader("People tend to post long review on music ")
        st.write(f'This graph shows number of words of posts.\
                 The average number of updating reviews in the \
                 dataset is {mean_length}.')
    "---"   

    st.subheader("Sample Posts")
    if st.button("Show Sample Posts and Sentiment"):
        placeholder = st.empty()
        with placeholder.container():
            for index, row in df.sample(3).iterrows():
                text = row["selftext"].strip()
                if text != "":
                    col1, col2 = st.columns([3,1])
                    with col1:
                        with st.expander(text[:100] + "..."):
                            st.write(text)
                    with col2:
                        if row["Liked"] == 1:
                            st.info(row['Liked'])
                        else:
                            st.error(row['Liked'])
        if st.button("Clear", type="primary"):
            placeholder.empty()
    #

with model:
    st.subheader("Our Sentiment Model")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "70%", "3%")
    col2.metric("F1 Score", "0.67", "0.03")
    with col3:
        st.write("We have use multiple ways to improve the performance of model. However, the performance doesn't \
                increase too much. The baseline model does the best work surprisingly. After we import spacy and use\
                lemma, pos, and dep features, the performance even gets worse...")
    
    st.subheader("Model Training Dataset")
    # load the training data file from bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket('trainingdata_pipeline')
    blob = bucket.blob('train.csv')
    data_file = tempfile.gettempdir() + '/music_reviews.csv'
    blob.download_to_filename(data_file)
    df_train = pd.read_csv(data_file)
    #make data balanced
    df_1=df_train[df_train.Liked==1]
    df_0=df_train[df_train.Liked==0]
    df_11=resample(df_1, replace= False, n_samples=df_0.shape[0],random_state=1)
    df_train=pd.concat([df_11, df_0])

    placeholder = st.empty()
    with placeholder.container():
        for index, row in df_train.sample(5).iterrows():
            text = row["Review"].strip()
            if text != "":
                col1, col2 = st.columns([3,1])
                with col1:
                    with st.expander(text[:100] + "..."):
                        st.write(text)
                with col2:
                    if row["Liked"] == 1:
                        st.info(row['Liked'])
                    else:
                        st.error(row['Liked'])
    "---"
    st.subheader("Model Training")
    pipeline_name=[str("pipeline1")
                    ,str("pipeline2")
                    ,str("pipeline3")]
    cv_score=[0.67, 0.66,0.681]
    improvement_describtion=[str("baseline of model")
                            ,str("decrease number of features")
                            ,str("import spacy to add more features")]
    accuracy=[0.67,0.67,0.63]
    dt_des=pd.DataFrame({"Pipeline Name":pipeline_name, 
                         "Cross Validation":cv_score, 
                         "Training Accuracy":accuracy, 
                         "Improvement Description":improvement_describtion})
    st.table(dt_des)
    "---"
    st.subheader("Model Improving")
    pipe1="""
    pipeline1 = Pipeline([
        ('vect', TfidfVectorizer(ngram_range=(1,3), stop_words='english', max_features=10000)),
        ('cls', LinearSVC(max_iter=120000))
    ])
    """

    pipe2="""
    pipeline2 = Pipeline([
        ('vect', TfidfVectorizer(ngram_range=(1,3), stop_words='english', max_features=2000)),
        ('cls', LinearSVC(max_iter=120000))
    ])
    """
    pipe3="""
    nlp = spacy.load('en_core_web_sm')
    def custom_tokenizer(document):
        doc_spacy = nlp(document)
        tokens=[]
        for token in doc_spacy:
            if (token.text.strip() != "") and (token.is_stop == False) and (token.is_alpha == True):
                tokens.append((token.lemma_ + '_' + token.pos_+'_'+token.tag_+'_'+token.dep_).lower())  
        #return [token for token in lemmas if token not in STOP_WORDS]
        return tokens
    pipeline3 = Pipeline([
            ('vect', TfidfVectorizer(tokenizer=custom_tokenizer, ngram_range=(1,3),max_features=2000)),
            ('cls', LinearSVC(max_iter=120000))
        ])
    """
    st.markdown("""
    - pipeline1
    """)
    st.code(pipe1,language='python')
    
    st.markdown("""
    - pipeline2
    """)
    st.code(pipe2,language='python')
    
    st.markdown("""
    - pipeline3
    """)
    st.code(pipe3,language='python')
    "---"
    
    st.subheader("Model Testing")
    pre_acc=[0.68,0.70,0.68]
    pre_f1=[0.68,0.70,0.67]
    pre_testing=pd.DataFrame({"Pipeline Name":pipeline_name, 
                         "Testing Accuracy":pre_acc, 
                         "Testing F1 score":pre_f1})
    st.table(pre_testing)
    "---"
    st.subheader("Final Model")
    f_pipe="""
        pipeline2 = Pipeline([
        ('vect', TfidfVectorizer(ngram_range=(1,3), stop_words='english', max_features=2000)),
        ('cls', LinearSVC(max_iter=120000))
    ])"""
    #show final pipeline code
    st.code(f_pipe
    ,language='python')
    "---"
    


with team:
    st.subheader("Team")
    "---"
    col1, col2 = st.columns([1,3])
    with col1:
        st.image("member.png")
    with col2:
        st.markdown(""" 
        Member Name:
        - Zhaofeng Liu
        - Jiacheng Dong
        - Qianhao Chen
        - Zhehao Jin
        - Tianle Shen
        """)
    
