from sklearn.feature_extraction.text import TfidfVectorizer

# Only 2 Hindi sentences
corpus = [
    "भारत एक महान देश है",
    "मशीन लर्निंग उपयोगी है"
]

#  (preprocessing)
processed = []
for sentence in corpus:
    processed.append(sentence.replace("है", ""))   # remove simple stopword

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed)