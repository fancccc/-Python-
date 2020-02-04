# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 23:37:16 2020

@author: Mario
"""
import pandas as pd
import math

def createDataSet_3():
    dataSet_3 = [
        # 1
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
        # 2
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
        # 3
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
        # 4
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
        # 5
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
        # 6
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
        # 7
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
        # 8
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],
        # 9
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
        # 10
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
        # 11
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
        # 12
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
        # 13
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
        # 14
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
        # 15
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
        # 16
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
        # 17
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']
    ]
    # 特征值列表
    labels_3 = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率','好坏']
    return dataSet_3, labels_3
a,b = createDataSet_3()
df = pd.DataFrame(a,columns = b)
#信息增益 
def Gain(D,C,classes):
    D['1'] = 1
    Ents = []
    groups = D.groupby(classes).count()
    groups[C] = groups[C] / sum(groups[C])
    groups['Ent_D'] = [i * math.log2(i) for i in groups[C]]
    Ent_D = -sum(groups['Ent_D'])
    Ents.append(Ent_D)
    if  type(D[C][0]) == str:#2 * len(set(D[C])) < len(D['1']):
        print('离散数据')
        groups = D.groupby(C)
        ent_d_sum = 0
        for name,group in groups:
            g = group.groupby(classes).count()
            g['p'] = g['1'] / sum(g['1'])
            g['ent'] = [i * math.log2(i) for i in g['p']]
            Ent_d = -sum(g['ent'])
            Ents.append(Ent_d)
            p = group.count()['1'] / sum(D['1'])
            ent_d_sum += p * Ent_d
        Gain = Ent_D - ent_d_sum
    else:
        print('连续数据')
        Gain = {}
        df = D.sort_values(C).reset_index()
        new = [(df[C][i] + df[C][i+1]) / 2 for i in range(len(df[C]) - 1)]
        for t in new:
            ent_d_sum = 0
            for group in [df[df[C] <= t],df[df[C] > t]]:
                g = group.groupby(classes).count()
                g['p'] = g['1'] / sum(g['1'])
                g['ent'] = [i * math.log2(i) for i in g['p']]
                Ent_d = -sum(g['ent'])
                Ents.append(Ent_d)
                p = group.count()['1'] / sum(D['1'])
                ent_d_sum += p * Ent_d
            Gain.update({Ent_D - ent_d_sum:t})               
        Gain = (max(list(Gain.keys())),Gain[max(list(Gain.keys()))])
    return Ents,Gain

for i in ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感','密度', '含糖率']:
    E,G = Gain(df,i,'好坏')
    print(i + '的信息增益为：',G)
'''
非连续数据
色泽的信息增益为： 0.10812516526536531
非连续数据
根蒂的信息增益为： 0.14267495956679277
非连续数据
敲击的信息增益为： 0.14078143361499584
非连续数据
纹理的信息增益为： 0.3805918973682686
非连续数据
脐部的信息增益为： 0.28915878284167895
非连续数据
触感的信息增益为： 0.006046489176565584
连续数据
密度的信息增益为： (0.2624392604045631, 0.3815)
连续数据
含糖率的信息增益为： (0.34929372233065203, 0.126)

纹理选作根结点划分 此后结点划分递归进行
'''
