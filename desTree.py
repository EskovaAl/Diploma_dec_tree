from collections import Counter

import numpy as np
from scipy.constants import value

from main import predictions


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
    def __init__(self, max_depth = 5, min_samples = 2):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.root = self.build_tree(X, y)

    # #Считаем неопределенность Джини
    # def gini_impurity(self, datas):
    #     gini_impur = np.array([1 - (pow(p, 2) + pow((1-p), 2)) for p in datas])
    #     print("Gini")
    #     return gini_impur
    #
    # #функция потерь - для Джини
    # def loss_func(self):
    #     print("loss_func")

    #Считаем энтропию
    def calc_entropy(self, y):

        hist = np.bincount(y)
        ps = hist / len(y)
        entr = - np.sum([p * np.log2(p) for p in ps if p>0])
        return entr

    #Информационный прирост - для энтропии
    def information_gain(self, left, right, parent):
        parent_entr = self.calc_entropy(parent)
        left_entr = self.calc_entropy(left)
        right_entr = self.calc_entropy(right)

        n = len(parent)
        n_left, n_right = len(left), len(right)
        gain = parent_entr - (n_left/n) * left_entr - (n_right/n) * right_entr
        print("Прирост", gain)
        return gain


    #Считаем ошибку классификации
    def class_error(self, datas):
        err = np.array([1 - max([p, 1-p]) for p in datas])
        print("MSE")
        return err

    #Вычисляем значение листа
    def calculate_leaf(self, y):
        #Классификация
        if self.n_classes == 2:
            values, counts = np.unique(y, return_counts=True)
            print("Калькуль", np.unique(y, return_counts=True))
            return values[np.argmax(counts)]
        else:
            #регрессия
            return np.mean(y)

    #ищем лучшее разбиение
    def best_split(self, X, y):

        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = 0
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
        return best_feature, best_threshold

    def build_tree(self, X, y, depth = 0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_samples < self.min_samples or n_labels == 1:
            print("Зашло в проверку окончания")
            leaf_value = self.calculate_leaf(y)
            return Node(value = leaf_value)
        best_feature, best_threshold = self.best_split(X, y)
        print("Бест", best_feature, best_threshold)
        if best_feature is None:
            print("Зашло в best_feature")
            leaf_value = self.calculate_leaf(y)
            return Node(value=leaf_value)

        left_indexes = np.argwhere(X[:, best_feature] <= best_threshold).flatten()
        right_indexes = np.argwhere(X[:, best_feature] > best_threshold).flatten()

        left_subtree = self.build_tree(X[left_indexes], y[left_indexes], depth + 1)
        right_subtree = self.build_tree(X[right_indexes], y[right_indexes], depth + 1)

        return Node(feature=best_feature, split_threshold=best_threshold, left=left_subtree, right=right_subtree)

    # #Строим дерево в рекурсии
    # def build_tree(self, X, y, depth = 0):
    #     n_samples, n_features = X.shape
    #     unique_classes = np.unique(y)
    #
    #     #проверка на лист
    #     if len(unique_classes) == 1 or (n_samples < self.min_samples) or (self.max_depth is not None and depth >= self.max_depth):
    #         leaf = self.calculate_leaf(y)
    #         return Node(value=leaf)
    #
    #     """
    #     ищем наилучшее разбиение
    #     best_feature - лучший признак
    #     best_threshold - наилучшее значение для разбиения
    #     """
    #     best_feature, best_threshold = self.best_split(X, y, n_features, n_samples)
    #
    #     left_indice = X[:, best_feature] < best_threshold
    #     right_indice = X[:, best_feature] >= best_threshold
    #     print(left_indice)
    #     left_subtree = self.build_tree(X[left_indice], y[left_indice], depth+1)
    #     right_subtree = self.build_tree(X[right_indice], y[right_indice], depth + 1)
    #
    #     return Node(feature = best_feature, split_threshold=best_threshold, left=left_subtree, right=right_subtree)

    def predict_input(self, x, tree):
        if tree.value is not None:
            return tree.value  # Лист

        if x[tree.feature] < tree.threshold:
            return self.predict_input(x, tree.left)
        else:
            return self.predict_input(x, tree.right)

    #Классифицируем новую инфу по уже построенному дереву
    def predict(self, X):
        print("classificate")
        return np.array([self.predict_input(x, self.root) for x in X])
