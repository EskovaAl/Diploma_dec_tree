'''
Вспомогательные функции
'''

import numpy as np

def extra_var(true_y, pred_y):
    '''
    true_y - y истинное
    pred_y - y предсказанное

    В функции рассчитываются дополнительные переменные для метрик
    TP - число истинно положительных
    TN - истинное отрицательные
    FP - ложно положительные
    FN - ложно отрицательные
    :return:
    '''
    tp = np.sum((true_y == 1) & (pred_y == 1))
    tn = np.sum((true_y == 0) & (pred_y == 0))
    fp = np.sum((true_y == 0) & (pred_y == 1))
    fn = np.sum((true_y == 1) & (pred_y == 0))
    return tp, tn, fp, fn

# Считаем ошибку классификации
# def class_error(self, true_y, pred_y):
#     err = np.array([1 - max([p, 1 - p]) for p in datas])
#     print("MSE")
#     return err
def accuracy(true_y, pred_y):
    #точность
    tp, tn, fp, fn = extra_var(true_y, pred_y)
    return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

def recall (true_y, pred_y):
    #полнота
    tp, tn, fp, fn = extra_var(true_y, pred_y)
    return tp/(tp+fn) if (tp+fn)>0 else 0

def precision(true_y, pred_y):
    #точность
    tp, tn, fp, fn = extra_var(true_y, pred_y)
    return tp/(tp+fp) if (tp+fn)>0 else 0

def f1_score(true_y, pred_y):
    prec = precision(true_y, pred_y)
    rec = recall(true_y, pred_y)
    return (2 * prec * rec)/(prec + rec) if (prec + rec)>0 else 0