from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd

from util import Util


class FeatureCreation:

    language = "english"

    def get_features(self, df, stop_word_exception):

        regex_pattern = "[^A-Za-z0-9 ]+"
        count_vect = CountVectorizer()
        tfidf_transformer = TfidfTransformer()
        util_obj = Util(language=self.language)

        # Lowering the comment
        df["comment_d"] = df["Failure"].apply(lambda comment: str.lower(comment))

        # Tokenizing the Comment
        df["token_list"] = df["comment_d"].apply(
            lambda comment: word_tokenize(comment, language=self.language))

        # Removing Stop Words
        df["cleaned_tokens"] = df["token_list"].apply(
            lambda token_list: util_obj.remove_stop_words(word_tokens=token_list,
                                                          exception_list=stop_word_exception))

        # Doing Stemming
        df["cleaned_tokens"] = df["cleaned_tokens"].apply(
            lambda token_list: util_obj.do_stemming(words=token_list))

        df["filtered_comment"] = df["cleaned_tokens"].apply(lambda tokens: " ".join(tokens))
        df["filtered_comment"] = df["filtered_comment"].apply(
            lambda str: util_obj.remove_regex(str, regex_pattern))
        # print(df)

        # Feature Engineering
        df_counts = count_vect.fit_transform(df["filtered_comment"])
        df_tfidf = tfidf_transformer.fit_transform(df_counts)
        tf_idf_df = pd.DataFrame(df_tfidf.toarray())

        df = df.join(tf_idf_df)

        # Doing One Hot Encoding for Mini-App
        df = util_obj.do_one_hot_encoding(df=df, col_name="Mini-App")
        return df
