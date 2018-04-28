from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class Model:

    def get_svm_model(self, X_train, y_train):
        clf = SVC()
        clf = clf.fit(X=X_train, y=y_train)
        return clf

    def svm_model(self, X_train, y_train, X_test, y_test):
        clf = self.get_svm_model(X_train, y_train)
        predicted = clf.predict(X_test)
        print(y_test.values)
        print(predicted)
        accuracy = np.mean(predicted == y_test)
        print("Accuracy :", accuracy)

    def get_knn_model(self, X_train, y_train):
        clf = KNeighborsClassifier()
        clf = clf.fit(X=X_train, y=y_train)
        return clf

    def knn_model(self, X_train, y_train, X_test, y_test):
        clf = self.get_knn_model(X_train, y_train)
        predicted = clf.predict(X_test)
        print(y_test.values)
        print(predicted)
        accuracy = np.mean(predicted == y_test)
        print("Accuracy :", accuracy)

    def get_dt_model(self, X_train, y_train):
        clf = DecisionTreeClassifier()
        clf = clf.fit(X=X_train, y=y_train)
        return clf

    def dt_model(self, X_train, y_train, X_test, y_test):
        clf = self.get_dt_model(X_train, y_train)
        predicted = clf.predict(X_test)
        print(y_test.values)
        print(predicted)
        accuracy = np.mean(predicted == y_test)
        print("Accuracy :", accuracy)

