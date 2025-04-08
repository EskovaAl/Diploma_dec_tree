'''
Вспомогательные функции
'''

import numpy as np


# Считаем ошибку классификации
# def class_error(self, true_y, pred_y):
#     err = np.array([1 - max([p, 1 - p]) for p in datas])
#     print("MSE")
#     return err

def accuracy(true_y, pred_y):
    #точность
    return np.mean(true_y==pred_y)

def recall (true_y, pred_y, class_label):
    #полнота для одного класса
    tp = np.sum((true_y == class_label) & (pred_y == class_label))
    fn = np.sum((true_y == class_label) & (pred_y != class_label))
    if tp + fn == 0:
        return 0.0
    else:
        return tp / (tp + fn)

def precision(true_y, pred_y, class_label):
    #точность для одного класса
    tp = np.sum((true_y == class_label) & (pred_y == class_label))
    fp = np.sum((true_y != class_label) & (pred_y == class_label))

    if tp + fp == 0:
        return 0.0
    else:
        return tp / (tp + fp)


def f1_score(true_y, pred_y, class_label):
    prec = precision(true_y, pred_y, class_label)
    rec = recall(true_y, pred_y, class_label)
    return (2 * prec * rec)/(prec + rec) if (prec + rec)>0 else 0

def confusion_matrix(true_y, pred_y, num_classes):
    #матрица ошибок
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(pred_y)):
        matrix[true_y[i], pred_y[i]] += 1
    return matrix