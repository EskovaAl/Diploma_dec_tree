import math
import numpy as np


class Node:
    """
    Узел в дереве.
    Атрибут - индекс атрибута, на основе которого разделяем
    value - значение атрибута
    left - левый узел
    right - правый узел
    split_threshold - граница разделения (значение порога)

    max_depth и min_samples вынести из Node в другой класс, здесь лишние.
    Создать класс Classification??? И внести туда все функции ниже???
    """
    def __init__(self, attribute_index = None, value = None, left = None, right = None, split_threshold = None):
        self.attribute_index = attribute_index
        self.value = value
        self.left = left
        self.right = right
        self.split_threshold = split_threshold

class DecisionTree:
    """
    
    """
    def __init__(self, max_depth = 50, min_samples = 2):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.root = None

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

    #ищем лучшее разбиение для Джини
    def best_split(self):
        print("Best split")


    #Строим дерево в рекурсии
    def build_tree(self):
        print("build_tree")

    #Классифицируем новую инфу по уже построенному дереву
    def classificate(self):
        print("classificate")