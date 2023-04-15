import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = {'text': ["I have seen a lot of movies...this is the first one I ever walked out of the theater on. Don't even bother renting it. This is about as boring a soap opera as one can see...at least you don't have to pay to watch a soap opera, though.",
"The Movie is okay. Meaning that I don't regret watching it! I found the acting purely and the most of the dialog stupid...",
"I work with children from 0 to 6 years old and they all love the Doodlebops...",
"I'm not sure why this film is averaging so low on IMDb when it's absolutely everything you could ever want in a horror film...",
"By reading the box at the video store this movie looks rather amusingly disturbing...",
"This movie has a few things going for it right off the bat...",
"Spider-Man is in my opinion the best superhero ever, and this game is the best superhero game ever...",
"A charming, funny film that gets a solid grade all around...",
"This movie was really stupid and I thought that it wasn't so bad and I could tolerate a movie about a bed eating people...",
"I'm giving this film 9 out of 10 only because there aren't enough specific scientific references to the amount of energy it takes to produce food to satisfy the science haters..."
]}
df = pd.DataFrame(data)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=20)
X = vectorizer.fit_transform(df['text'])

print("Feature names:")
print(vectorizer.get_feature_names())

print("\nFeature representation matrix:")
print(X.toarray())