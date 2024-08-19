import wfdb
import neurokit2 as nk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import Counter
import matplotlib.pyplot as plt
class Record:
    def __init__(self, record, N, V, F, O, L, R, A, a, J, S, V2, F2, O2, Q):
        self.record = record
        self.N = N
        self.V = V
        self.F = F
        self.O = O
        self.L = L
        self.R = R
        self.A = A
        self.a = a
        self.J = J
        self.S = S
        self.V2 = V2
        self.F2 = F2
        self.O2 = O2
        self.Q = Q

# Lista de objetos Record
records = [
    Record(100, 2239, None, None, 33, None, None, None, 1, None, None, None, None, None, None, None),
    Record(101, 1860, None, None, 3, None, None, None, None, None, None, None, None, None, None, 2),
    Record(102, 99, None, None, None, None, None, None, 4, None, None, None, None, 2028, 56, None),
    Record(103, 2082, None, None, 2, None, None, None, None, None, None, None, None, None, None, None),
    Record(104, 163, None, None, None, None, None, None, 2, None, None, None, None, 1380, 666, 18),
    Record(105, 2526, None, None, None, None, None, None, 41, None, None, None, None, None, None, 5),
    Record(106, 1507, None, None, None, None, None, None, 520, None, None, None, None, None, None, None),
    Record(107, None, None, None, None, None, None, None, 59, None, None, None, None, 2078, None, None),
    Record(108, 1739, None, None, 4, None, None, None, 17, 2, None, None, 1, None, None, 11),
    Record(109, None, 2492, None, None, None, None, None, 38, 2, None, None, None, None, None, None),
    Record(111, None, 2123, None, None, None, None, None, 1, None, None, None, None, None, None, None),
    Record(112, 2537, None, None, 2, None, None, None, None, None, None, None, None, None, None, None),
    Record(113, 1789, None, None, None, 6, None, None, None, None, None, None, None, None, None, None),
    Record(114, 1820, None, None, 10, None, 2, None, 43, 4, None, None, None, None, None, None),
    Record(115, 1953, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
    Record(116, 2302, None, None, 1, None, None, None, 109, None, None, None, None, None, None, None),
    Record(117, 1534, None, None, 1, None, None, None, None, None, None, None, None, None, None, None),
    Record(118, None, None, 2166, 96, None, None, None, 16, None, None, None, None, None, 10, None),
    Record(119, 1543, None, None, None, None, None, None, 444, None, None, None, None, None, None, None),
    Record(121, 1861, None, None, 1, None, None, None, 1, None, None, None, None, None, None, None),
    Record(122, 2476, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
    Record(123, 1515, None, None, None, None, None, None, 3, None, None, None, None, None, None, None),
    Record(124, None, None, 1531, 2, None, 29, None, 47, 5, None, None, 5, None, None, None),
    Record(200, 1743, None, None, 30, None, None, None, 826, 2, None, None, None, None, None, None),
    Record(201, 1625, None, None, 30, 97, 1, None, 198, 2, None, None, 10, None, None, 37),
    Record(202, 2061, None, None, 36, 19, None, None, 19, 1, None, None, None, None, None, None),
    Record(203, 2529, None, None, None, 2, None, None, 444, 1, None, None, None, None, None, 4),
    Record(205, 2571, None, None, 3, None, None, None, 71, 11, None, None, None, None, None, None),
    Record(207, None, 1457, 86, 107, None, None, None, 105, None, 472, None, None, 105, None, None),
    Record(208, 1586, None, None, None, None, None, 2, 992, 373, None, None, None, None, None, 2),
    Record(209, 2621, None, None, 383, None, None, None, 1, None, None, None, None, None, None, None),
    Record(210, 2423, None, None, None, 22, None, None, 194, 10, None, None, 1, None, None, None),
    Record(212, 923, None, 1825, None, None, None, None, None, None, None, None, None, None, None, None),
    Record(213, 2641, None, None, 25, 3, None, None, 220, 362, None, None, None, None, None, None),
    Record(214, None, 2003, None, None, None, None, None, 256, 1, None, None, None, None, None, 2),
    Record(215, 3195, None, None, 3, None, None, None, 164, 1, None, None, None, None, None, None),
    Record(217, 244, None, None, None, None, None, None, 162, None, None, None, None, 1542, 260, None),
    Record(219, 2082, None, None, 7, None, None, None, 64, 1, None, None, None, None, 133, None),
    Record(220, 1954, None, None, 94, None, None, None, None, None, None, None, None, None, None, None),
    Record(221, 2031, None, None, None, None, None, None, 396, None, None, None, None, None, None, None),
    Record(222, 2062, None, None, 208, None, 1, None, None, None, None, None, 212, None, None, None),
    Record(223, 2029, None, None, 72, 1, None, None, 473, 14, None, 16, None, None, None, None),
    Record(228, 1688, None, None, 3, None, None, None, 362, None, None, None, None, None, None, None),
    Record(230, 2255, None, None, None, None, None, None, 1, None, None, None, None, None, None, None),
    Record(231, 314, None, 1254, 1, None, None, None, 2, None, None, None, None, None, None, 2),
    Record(232, None, None, 397, 1382, None, None, None, None, None, None, 1, None, None, None, None),
    Record(233, 2230, None, None, 7, None, None, None, 831, 11, None, None, None, None, None, None),
    Record(234, 2700, None, None, None, None, 50, None, 3, None, None, None, None, None, None, None)
]

X=[]
y=[]

