import math
import numpy as np
from numpy.ma.core import left_shift


class Node:
    """
    Узел в дереве.
    Атрибут - индекс атрибута, на основе которого разделяем
    left - левый узел
    right - правый узел
    result - граница разделения (значение порога)

    max_depth и min_samples вынести из Node в другой класс, здесь лишние.
    Создать класс Classification??? И внести туда все функции ниже???
    """
    def __init__(self, attribute = None, left = None, right = None, result = None):
        self.attribute = attribute
        self.left = left
        self.right = right
        self.result = result
        #self.max_depth
        #self.min_samples

#Считаем неопределенность Джини
def gini_impurity(datas):
    gini_impur = np.array([1 - (pow(p, 2) + pow((1-p), 2)) for p in datas])
    print("Gini")
    return gini_impur

#Считаем энтропию
def entropy(datas):
    entr = np.array([-1 * (p * np.log2(p) + (1-p) + np.log2(1-p)) for p in datas])
    print("entropy")
    return entr

#Считаем ошибку классификации
def class_error(datas):
    err = np.array([1 - max([p, 1-p]) for p in datas])
    print("MSE")
    return err

#функция потерь - для Джини
def loss_func():
    print("loss_func")

#Информационный прирост - для энтропии
def information_gain(datas, attr_index):

    print("information_gain")

#ищем лучшее разбиение для Джини
def best_split():
    print("Best split")


#Строим дерево в рекурсии
def build_tree():
    print("build_tree")

#Классифицируем новую инфу по уже построенному дереву
def classificate():
    print("classificate")