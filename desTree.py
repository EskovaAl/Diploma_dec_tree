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
    def __init__(self, feature = None, value = None, left = None, right = None, split_threshold = None):
        print("зашли в инит узла")
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.split_threshold = split_threshold
        print("значение узла", self.value)

class DecisionTree:
    """
    min_samples - Минимальное количество семплов
    max_depth - максимальная глубина
    """
    def __init__(self, max_depth=float('inf'), min_samples = 2):

        self.min_samples = min_samples
        self.max_depth = max_depth
        print(f"зашли в init дерева, глубина {max_depth}, в селф {self.max_depth}, сэмплы {min_samples}, в селф {self.min_samples}")

    def fit(self, X, y):
        print("зашли в fit")
        self.n_classes = len(set(y))  # Количество классов
        self.n_features = X.shape[1]
        self.root = self.build_tree(X, y)
        print(self.root)

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
        gain = parent_entr - ((n_left/n) * left_entr - (n_right/n) * right_entr)
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
        most_common = np.bincount(y).argmax()
        return most_common

    #ищем лучшее разбиение
    def best_split(self, X, y, n_features):

        best_feature = None
        best_threshold = None
        best_gain = -1

        for feature in range(n_features):
            thresholds = np.unique(X[:,feature])

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
        print(f' зашли в билд три')
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_samples < self.min_samples or n_labels == 1:
            print("Зашло в проверку окончания")
            leaf_value = self.calculate_leaf(y)
            return Node(value = leaf_value)

        #Ищем лучшее разделение
        best_feature, best_threshold = self.best_split(X, y, n_features)
        print("Бест", best_feature, best_threshold)

        #Если не найдено подходящее разделение
        if best_feature is None:
            print("Зашло в best_feature")
            leaf_value = self.calculate_leaf(y)
            return Node(value=leaf_value)

        #Разделяем на два поддерева
        # left_indexes = np.argwhere(X[:, best_feature] <= best_threshold).flatten()
        # right_indexes = np.argwhere(X[:, best_feature] > best_threshold).flatten()

        left_indexes = X[:, best_feature] < best_threshold
        right_indexes = X[:, best_feature] >= best_threshold
        left_subtree = self.build_tree(X[left_indexes], y[left_indexes], depth + 1)
        right_subtree = self.build_tree(X[right_indexes], y[right_indexes], depth + 1)

        return Node(feature=best_feature, split_threshold=best_threshold, left=left_subtree, right=right_subtree)

    def predict_input(self, x):
        tree = self.root
        print(x)
        while tree.value is None:
            print("лучшее в предсказании", tree.split_threshold, x[tree.feature])
            if x[tree.feature] < tree.split_threshold:
                tree = tree.left
            else:
                tree = tree.right
        return tree.value

    #Классифицируем новую инфу по уже построенному дереву
    def predict(self, X):
        print("Классифицируем новые данные")
        return np.array([self.predict_input(x) for x in X])
