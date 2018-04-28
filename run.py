import pandas as pd
from util import Util
from feature_creation import FeatureCreation
from sklearn.model_selection import train_test_split

from model import Model


file_path = "data/SampleData.csv"
stop_word_exception = ["not"]
util_obj = Util()
fe = FeatureCreation()
model = Model()

df = pd.read_csv(file_path)

df = fe.get_features(df=df, stop_word_exception=stop_word_exception)
# df.to_csv("data/out.csv")
train, test = train_test_split(df, test_size=0.3)

print("Train Data")
print(train)

print("Test Data")
print(test)

columns_to_remove = ["Mini-App", "Script ID", "Failure", "Reason", "comment_d", "token_list", "cleaned_tokens", "filtered_comment"]
train_columns = list(set(df.columns) - set(columns_to_remove))
label_column = "Reason"

X_train = train.loc[:, train.columns.isin(train_columns)]
y_train = train[label_column]

X_test = test.loc[:, test.columns.isin(train_columns)]
y_test = test[label_column]

print("Training Data Details")
print(X_train.shape)

print("Test Data Details")
print(X_test.shape)

print("\nDecision Tree Classifier")
model.dt_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

print("\nKNN Classifier")
model.knn_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

print("\nSVM Classifier")
model.svm_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)








