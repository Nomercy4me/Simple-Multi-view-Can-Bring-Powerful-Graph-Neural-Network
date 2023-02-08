#-*- coding:UTF-8 -*-
# 评价指标
# presicion@k recall@k f1@k MAP DCG IDCG NDCG

import math
import numpy as np
import tensorflow as tf

def label_alignment(output,label,mask,sample_num,class_num):
    label = np.squeeze(label)
    mask = np.squeeze(mask)
    # 对预测结果降序排列
    index = np.argsort(-output, axis=1)
    # 把label和预测结果的降序排列对齐
    label_align = np.zeros(label.shape)

    for i in range(sample_num):
        if (mask[i] == False):
            continue
        for j in range(class_num):
            label_align[i, j] = label[i, index[i, j]]
    return label_align

def precision_at_K(label_align, mask,k,sample_num):
    mask = np.squeeze(mask)
    precision = []
    for i in range(sample_num):
        if(mask[i] == False):
            continue
        precision.append(sum(label_align[i,:k])/float(k))
    averge_precision_at_k = sum(precision)/len(precision)
    print "precision: ",averge_precision_at_k
    return averge_precision_at_k

def recall_at_K(label_align,label,mask,k,sample_num):
    label = np.squeeze(label)
    mask = np.squeeze(mask)
    recall = []
    for i in range(sample_num):
        if(mask[i] == False):
            continue
        recall.append(sum(label_align[i,:k])/float(sum(label[i])))
    averge_recall_at_k = sum(recall) / len(recall)
    print "recall: ",averge_recall_at_k
    return averge_recall_at_k

def F1score_at_K(label_align,label,mask,k,sample_num):
    label = np.squeeze(label)
    mask = np.squeeze(mask)
    # 2*p*r/(p+r)
    precision = []
    for i in range(sample_num):
        if (mask[i] == False):
            continue
        precision.append(sum(label_align[i, :k]) / float(k))
    recall = []
    for i in range(sample_num):
        if (mask[i] == False):
            continue
        recall.append(sum(label_align[i, :k]) / float(sum(label[i])))
    f1 = []
    for i in range(len(precision)):
        if(precision[i]+recall[i] == 0):
            f1.append(0)
        else:
            f1.append(2*precision[i]*recall[i]/(precision[i]+recall[i]))
    averge_f1_at_k = sum(f1) / len(f1)
    print "f1: ",averge_f1_at_k
    return averge_f1_at_k

def MAP(label_align,mask,sample_num,class_num):
    # 每个真实label出现时的precision_k的加和再除以label个数
    # 所有样本求均值
    mask = np.squeeze(mask)
    map_value = []
    for i in range(sample_num):
        if(mask[i] == False):
            continue
        map_sum = 0
        for j in range(class_num):
            if(label_align[i,j]==1):
                map_sum += sum(label_align[i,:(j+1)])/float(j+1)

        map_sum = map_sum/sum(label_align[i])
        map_value.append(map_sum)

    average_map = sum(map_value)/len(map_value)
    print "map",average_map
    return average_map

def ndcg(label,label_align,mask,k,sample_num):
    label = np.squeeze(label)
    mask = np.squeeze(mask)
    ndcg_value = []
    for i in range(sample_num):
        if(mask[i]== False):
            continue
        dcg_sum = 0
        for j in range(k):
            dcg_sum += label_align[i,j]/(math.log(j+2,2))
        idcg_sum = 0
        label_num = sum(label[i])
        for j in range(int(min(k,label_num))):
            idcg_sum += 1.0/(math.log(j+2,2))

        ndcg_value.append(dcg_sum/idcg_sum)

    average_ndcg_at_k = sum(ndcg_value) / len(ndcg_value)
    print "ndcg: ",average_ndcg_at_k
    return average_ndcg_at_k

if __name__== "__main__":
    output = np.array([[0.3, 0.6, 0.2], [0.2, 0.7, 0.6], [1.0, 3.0, 2.0]])
    label = np.array([[1, 0, 1], [0, 1, 1], [0, 1, 0]])
    mask = [True, False, True]
    k = 2

    sample_num = label.shape[0]
    class_num = label.shape[1]

    label_align = label_alignment(output,label,mask,sample_num,class_num)
    precision_at_K(label_align, mask,k,sample_num)
    recall_at_K(label_align,label,mask,k,sample_num)
    F1score_at_K(label_align,label,mask,k,sample_num)
    MAP(label_align,mask,sample_num,class_num)
    ndcg(label,label_align,mask,k,sample_num)