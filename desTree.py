from collections import Counter
import numpy as np
from scipy.spatial import distance


class Node:
    """
    Узел в дереве.
    feature - Атрибут, на основе которого разделяем(ИНДЕКС)
    left - левый узел
    right - правый узел
    split_threshold - граница разделения (значение порога)
    X_leaf, y_leaf - для храния объектов в листе
    is_leaf - для идентификации лист узел или нет
    """

    def __init__(self, feature=None, split_threshold=None, left=None, right=None, X_leaf=None, y_leaf=None, is_leaf=False):
        self.feature = feature
        self.split_threshold = split_threshold
        self.left = left
        self.right = right
        self.X_leaf = X_leaf
        self.y_leaf = y_leaf
        self.is_leaf = is_leaf



class DecisionTree:
    """
    min_samples_split - Минимальное количество семплов для разбиения
    max_depth - максимальная глубина
    knn_neighbor - число соседей
    """

    def __init__(self, min_samples_split=2, max_depth=float('inf'), crit = "gini", knn_neighbor =3):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.crit = crit
        self.knn_neighbor = knn_neighbor

    def fit(self, X, y):
        #Обучение
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.root = self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):
        #Рекурсивно строим дерево
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        #Проверяем окончание
        if (n_samples < self.min_samples_split) or (depth >= self.max_depth) or (n_labels == 1):
            #Сичтаем лист на основании наиболее популярного класса
            return Node(is_leaf = True, X_leaf=X, y_leaf=y)

        #Ищем наилучшее разделение данных
        best_feature, best_threshold = self.best_split(X, y, n_features)
        print("Бест", best_feature, best_threshold)

        if best_feature is None:  #если нет подходящего разделения
            return Node(is_leaf = True, X_leaf=X, y_leaf=y)

        #Разделение данных на подмножества
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold
        left_node = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right_node = self.build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, split_threshold=best_threshold, left=left_node, right=right_node)

    def best_split(self, X, y, n_features):
        #функция для поиска наилучшего разделения
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                gain = self.information_gain(y, y[left_indices], y[right_indices])
                print(gain)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def information_gain(self, parent, left_child, right_child):
        #Прирост информации
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        if self.crit == 'entropy':
            gain = self.entropy(parent) - (weight_left * self.entropy(left_child) + weight_right * self.entropy(right_child))
        elif self.crit == 'gini':
            gain = self.gini_index(parent) - (
                        weight_left * self.gini_index(left_child) + weight_right * self.gini_index(right_child))
        else:
            raise ValueError("Такого критерия нет")
        return gain

    def gini_index(self, y):
        if len(y) == 0:
            return 0
        class_count = Counter(y)
        return 1 - sum((count/len(y)) ** 2 for count in class_count.values())

    # def gini_index(self, y):
    #     class_labels, counts = np.unique(y, return_counts=True)
    #     probabilities = counts / counts.sum()
    #     return 1 - np.sum(probabilities ** 2)

    def entropy(self, y):
        #Считаем энтропию
        class_labels, hist = np.unique(y, return_counts=True)
        print("counts", hist)
        ps = hist / hist.sum()
        print('ps', ps)
        entr = -np.sum(ps * np.log2(ps + 1e-9)) #фикс для деления на ноль
        return entr
    #
    # def calculate_leaf(self, y):
    #     #функция расчета значения листа
    #     most_common = np.bincount(y).argmax()
    #     return most_common


    def knn_predict(self, x, X_leaf, y_leaf):
        '''
        предсказываем метку класса используя knn
        :param x:выборка
        :param X_leaf: данные Х в узле
        :param y_leaf: данные у в узле
        :return:
        '''
        dist = [distance.euclidean(x, x_leaf) for x_leaf in X_leaf]
        k_near_indices = np.argsort(dist)[:self.knn_neighbor]
        k_nearest_labels = y_leaf[k_near_indices]
        most_common  = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


    def predict(self, X):
        return [self.predict_input(x) for x in X]

    def predict_input(self, x):
        node = self.root
        while not node.is_leaf:  # Пока не достигнут листовой узел
            if x[node.feature] < node.split_threshold:
                node = node.left
            else:
                node = node.right
        return self.knn_predict(x, node.X_leaf, node.y_leaf)


