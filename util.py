from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import re


class Util:

    def __init__(self, language="english"):
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()

    def remove_stop_words(self, word_tokens, exception_list):
        stop_words = list(set(self.stop_words) - set(exception_list))
        filtered_tokens = [w for w in word_tokens if not w in stop_words]
        return filtered_tokens

    def do_stemming(self, words):
        out = []
        for word in words:
            out.append(self.stemmer.stem(word))
        return out

    def do_one_hot_encoding(self, df, col_name):
        one_hot = pd.get_dummies(df[col_name])
        df = df.join(one_hot)
        return df

    def remove_regex(self, input_text, regex_pattern):
        # urls = re.finditer(regex_pattern, input_text)
        # for i in urls:
        #     input_text = re.sub(i.group().strip(), '', input_text)
        input_text = re.sub(regex_pattern, '', input_text)
        return input_text

