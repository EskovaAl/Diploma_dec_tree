import math
import numpy as np
from pyexpat import features

from numpy.ma.extras import unique


class Node:
    """
    Узел в дереве.
    feature - Атрибут, на основе которого разделяем
    value - значение атрибута
    left - левый узел
    right - правый узел
    split_threshold - граница разделения (значение порога)
    """
    def __init__(self, feature = None, value = None, left = None, right = None, split_threshold = None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.split_threshold = split_threshold

class DecisionTree:
    """
    min_samples - Минимальное количество семплов
    max_depth - максимальная глубина
    root - корень
    """
    def __init__(self, max_depth = None, min_samples = 2):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.root = self.build_tree()

    #Считаем неопределенность Джини
    def gini_impurity(self, datas):
        gini_impur = np.array([1 - (pow(p, 2) + pow((1-p), 2)) for p in datas])
        print("Gini")
        return gini_impur

    #Считаем энтропию
    def entropy(self, datas):
        entr = np.array([-1 * (p * np.log2(p) + (1-p) + np.log2(1-p)) for p in datas])
        print("entropy")
        return entr

    #Считаем ошибку классификации
    def class_error(self, datas):
        err = np.array([1 - max([p, 1-p]) for p in datas])
        print("MSE")
        return err

    #функция потерь - для Джини
    def loss_func(self):
        print("loss_func")

    #Информационный прирост - для энтропии
    def information_gain(self, datas, attr_index):

        print("information_gain")

    #ищем лучшее разбиение
    def best_split(self, X, y, n_features, n_samples):

        best_feature = None
        best_threshold = None
        best_gain = float("-inf")

        for feature in range(n_features):
            thresholds = np.unique(X[:feature])

            for threshold in thresholds:
                left_indice = X[:, feature] < threshold
                right_indice = X[:, feature] >= threshold

                if len(y[left_indice]) == 0 or len(y[right_indice]) == 0:
                    continue
                gain = self.information_gain(y, y[left_indice], y[right_indice])
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
                    best_feature = feature
        print("Best split")


    #Строим дерево в рекурсии
    def build_tree(self, X, y, depth = 0):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        #проверка на лист
        if len(unique_classes == 1) or (n_samples < self.min_samples_split) or (self.max_depth is not None and depth >= self.max_depth):
            leaf = self.calculate_leaf(y)
            return Node(value=leaf)

        """
        ищем наилучшее разбиение
        best_feature - лучший признак
        best_threshold - наилучшее значение для разбиения
        """
        best_feature, best_threshold = self.best_split(X, y, n_features, n_samples)

        left_indice = X[:, best_feature] < best_threshold
        right_indice = X[:, best_feature] >= best_threshold
        left_subtree = self.build_tree(X[left_indice], y[left_indice], depth+1)
        right_subtree = self.build_tree(X[right_indice], y[right_indice], depth + 1)

        return Node(feature = best_feature, split_threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _predict_input(self, x, tree):
        if tree.value is not None:
            return tree.value  # Лист

        if x[tree.feature] < tree.threshold:
            return self._predict_input(x, tree.left)
        else:
            return self._predict_input(x, tree.right)

    #Классифицируем новую инфу по уже построенному дереву
    def predict(self, X):
        print("classificate")
        return np.array([self._predict_input(x, self.root) for x in X])