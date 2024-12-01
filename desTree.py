import math
import numpy as np


class Node:
    def __init__(self, attribute = None, value = None, children = None, result = None):
        self.attribute = attribute
        self.value = value
        self.children = children
        self.result = result
        #self.max_depth
        #self.min_samples

#Считаем неопределнность Джини
def gini(datas):
    gini_neopr = np.array([1 - (pow(p, 2) + pow((1-p), 2)) for p in datas])
    print("Jini")
    return gini_neopr

#Считаем энтропию
def entropy(datas):
    entr = np.array([-1 * (p * np.log2(p) + (1-p) + np.log2(1-p)) for p in datas])
    print("entropy")
    return entr

#Считаем ошибку классификации
def class_error(datas):
    err = np.array([1 - max([p, 1-p]) for p in datas])
    print("entropy")
    return err

#функция потерь - для Джини
def loss_func():
    print("loss_func")

#Информационный прирост - для энтропии
def information_gain():
    print("information_gain")

#Строим дерево
def build_tree():
    print("build_tree")

#Классивицируем новую инфу по уже построенному дереву
def classificate():
    print("classificate")