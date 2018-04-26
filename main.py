import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from util import Util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

file_path = "/media/manish/Data1/personal/project/gaurav-project/data/GSCHData-Dummy.csv"

data = pd.read_csv(file_path)
language = "english"
stop_word_exception = ["not"]

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()
util_obj = Util(language=language)


# Lowering the comment
data["comment_d"] = data["Comment"].apply(lambda comment: str.lower(comment))

# Tokenizing the Comment
data["token_list"] = data["comment_d"].apply(lambda comment: word_tokenize(comment, language=language))

# Removing Stop Words
data["cleaned_tokens"] = data["token_list"].apply(
    lambda token_list: util_obj.remove_stop_words(word_tokens=token_list,
                                                  exception_list=stop_word_exception))

# Doing Stemming
data["cleaned_tokens"] = data["cleaned_tokens"].apply(
    lambda token_list: util_obj.do_stemming(words=token_list))

data["filtered_comment"] = data["cleaned_tokens"].apply(lambda tokens: " ".join(tokens))
# print(data)

# Feature Engineering
X_train_counts = count_vect.fit_transform(data["filtered_comment"])
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Steps line no 20 - 40 should also be done for test data.

# Modelling

# Model will be trained only on Train Data
clf = MultinomialNB().fit(X_train_tfidf, data["Defect Type"])


predicted = clf.predict(X_train_tfidf) # Modify X_train_tfidf with your test data frame after applying step of line 20-40.
accuracy = np.mean(predicted == data["Defect Type"])
print("Accuracy :", accuracy)




