from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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
