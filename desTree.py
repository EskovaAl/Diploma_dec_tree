import numpy as np


class Node:
    """
    Узел в дереве.
    feature - Атрибут, на основе которого разделяем
    value - значение атрибута
    left - левый узел
    right - правый узел
    split_threshold - граница разделения (значение порога)
    """

    def __init__(self, feature=None, split_threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.split_threshold = split_threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    """
    min_samples - Минимальное количество семплов
    max_depth - максимальная глубина
    """

    def __init__(self, min_samples=2, max_depth=float('inf')):
        self.min_samples = min_samples
        self.max_depth = max_depth

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
        if (n_samples < self.min_samples) or (depth >= self.max_depth) or (n_labels == 1):
            leaf_value = self.calculate_leaf(y)  #Сичтаем лист на основании наиболее популярного класса
            return Node(value=leaf_value)

        #Ищем наилучшее разделение данных
        best_feature, best_threshold = self.best_split(X, y, n_features)
        print("Бест", best_feature, best_threshold)

        if best_feature is None:  # Если нет подходящего разделения
            leaf_value = self.calculate_leaf(y)
            return Node(value=leaf_value)

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
        gain = self.entropy(parent) - (
                    weight_left * self.entropy(left_child) + weight_right * self.entropy(right_child))
        return gain

    def entropy(self, y):
        #Считаем энтропию
        class_labels, hist = np.unique(y, return_counts=True)
        print("counts", hist)
        ps = hist / hist.sum()
        print('ps', ps)
        entr = -np.sum(ps * np.log2(ps + 1e-9))  # Добавляем малую величину для избежания деления на ноль
        return entr

    # def gini_index(self, y):
    #     class_labels, counts = np.unique(y, return_counts=True)
    #     probabilities = counts / counts.sum()
    #     return 1 - np.sum(probabilities ** 2)

    def calculate_leaf(self, y):
        #Рассчитываем лист
        most_common = np.bincount(y).argmax()
        return most_common

    def predict(self, X):
        return np.array([self.predict_input(sample) for sample in X])

    def predict_input(self, sample):
        node = self.root
        while node.value is None:  # Пока не достигнут листовой узел
            if sample[node.feature] < node.split_threshold:
                node = node.left
            else:
                node = node.right
        return node.value

