from data.env_dataset_support import *
import numpy as np
import matplotlib.pyplot as plt
import os
import random



if __name__ == "__main__":
    file_path = '/home/siyao/project/rlPractice/MiniGrid/generator/data/grid10000.pkl'
    dict = load_dic(file_path)
    cnt = len(dict)
    feature_map = np.zeros((cnt, 2))
    idx = 0
    for key in dict:
        feature_map[idx, 0] = key[0]/1000
        feature_map[idx, 1] = key[1]/1000
        idx += 1
    plt.scatter(feature_map[:, 0], feature_map[:, 1])
    plt.xlabel('empty_ratio')
    plt.ylabel('path_length')
    plt.title('Feature Map')
    plt.show()
    pass