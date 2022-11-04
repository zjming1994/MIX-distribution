# -*- coding: utf-8 -*-

"""
测试等比例混合的比例

@author: ZJM
"""

import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


# 二维柱状图格子频数
# size是这个方法唯一的参数：每一维设定size个柱子
def find_all(data, xy, size):
    Z = np.zeros((size, size))
    x = np.linspace(xy[0], xy[1], size+1)
    y = np.linspace(xy[2], xy[3], size+1)
    for i in range(size):
        for j in range(size):
            down = np.array([[x[i], y[j]]]) < data.iloc[:,[0,2]]
            upper = np.array([[x[i+1], y[j+1]]]) >= data.iloc[:,[0,2]]
            Z[i,j] = np.sum(np.all(down & upper, axis=1))
            Z = Z.astype(int)
    return Z


# JS散度函数
def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p,M)+0.5*scipy.stats.entropy(q, M)


def get_pro_JS(df_A, df_B, df_AB, size, xy):
    # 格子拉直
    print('正在计算A的经验分布...')
    Z1 = find_all(df_A, xy, size)
    print('正在计算B的经验分布...')
    Z2 = find_all(df_B, xy, size)
    z1 = Z1.ravel()/np.sum(Z1)
    z2 = Z2.ravel()/np.sum(Z2)
    print('正在计算A和B混合的经验分布...')
    C = find_all(df_AB, xy, size)   #混合样本
    print('经验分布计算结束！')
    c = C.ravel()/np.sum(C)
    # 网格搜索最优JS散度和混合值,保留4位有效数字
    w_a = np.linspace(0,1,10001)
    js_list = []
    print('正在进行网格搜索...')
    for i in w_a:
        js = JS_divergence(i*z1+(1-i)*z2, c)
        js_list.append(js)
    js_a = np.array(js_list)
    ans = (np.vstack((w_a, js_a))).T
    print('网格搜索结束！')
    return ans


def get_answer(ans):
    res = ans[ans[:,1] == np.min(ans[:,1])]
    pro = res[0, 0]
    best_js = res[0, 1]
    print('A的权重为[%.4f]时，对应JS散度为[%.6f]'%(pro, best_js))
    return pro, best_js


def get_test_Datas(path_A, path_B, path_AB):
    # 数据读入
    df_A = pd.read_excel(path_A)    # 分子A的数据
    df_B = pd.read_excel(path_B)    # 分子B的数据
    df_AB = pd.read_excel(path_AB)    # 混合分子的数据（真实）
    df_AB = df_AB.astype('float')
    # 第2列数据取对数
    df_A['logY'] = np.log10(df_A.iloc[:,1])
    df_B['logY'] = np.log10(df_B.iloc[:,1])
    df_AB['logY'] = np.log10(df_AB.iloc[:,1])
    # 划定格子范围,因闭区间需要,左右放宽1e-7
    # x的最小,x的最大,y的最小,y的最大
    xy = [min(np.min(df_A, axis = 0)[0], np.min(df_B, axis = 0)[0])-1e-7,
          max(np.max(df_A, axis = 0)[0], np.max(df_B, axis = 0)[0])+1e-7,
          min(np.min(df_A, axis = 0)[2], np.min(df_B, axis = 0)[2])-1e-7,
          max(np.max(df_A, axis = 0)[2], np.max(df_B, axis = 0)[2])+1e-7]
    return df_A, df_B, df_AB, xy


def get_train_Datas(path_A, path_B, size_train, size_A, size_B):
    # 数据读入
    df_A = pd.read_excel(path_A)    # 分子A的数据
    df_B = pd.read_excel(path_B)    # 分子B的数据
    # 第2列数据取对数
    df_A['logY'] = np.log10(df_A.iloc[:,1])
    df_B['logY'] = np.log10(df_B.iloc[:,1])
    indexA = np.random.choice(a=df_A.shape[0], size=size_train, replace=False)
    indexB = np.random.choice(a=df_B.shape[0], size=size_train, replace=False)
    train_A = df_A.iloc[indexA]
    train_B = df_B.iloc[indexB]
    # 划定格子范围,因闭区间需要,左右放宽1e-7
    # x的最小,x的最大,y的最小,y的最大
    xy = [min(np.min(train_A, axis = 0)[0], np.min(train_B, axis = 0)[0])-1e-7,
          max(np.max(train_A, axis = 0)[0], np.max(train_B, axis = 0)[0])+1e-7,
          min(np.min(train_A, axis = 0)[2], np.min(train_B, axis = 0)[2])-1e-7,
          max(np.max(train_A, axis = 0)[2], np.max(train_B, axis = 0)[2])+1e-7]
    index1 = np.random.choice(a=df_A.shape[0], size=size_A, replace=False)
    index2 = np.random.choice(a=df_B.shape[0], size=size_B, replace=False)
    train_AB = pd.concat([df_A.iloc[index1], df_B.iloc[index2]])
    return train_A, train_B, train_AB, xy


def main_draw():
    path_A = '6SLP--39240 events.xlsx'
    path_B = '3SLP--34595 events(1).xlsx'
    size_train = 30000
    size = 20  # 柱子的个数

    # 2:8
    size_A = 7500
    size_B = 30000
    train_A, train_B, train_AB, xy = get_train_Datas(path_A, path_B, size_train, size_A, size_B)
    answer1 = get_pro_JS(train_A, train_B, train_AB, size, xy)
    pro1, best_js1 = get_answer(answer1)
    print('-'*50, '2:8', '-'*50)

    # 4:6
    size_A = 20000
    size_B = 30000
    train_A, train_B, train_AB, xy = get_train_Datas(path_A, path_B, size_train, size_A, size_B)
    answer2 = get_pro_JS(train_A, train_B, train_AB, size, xy)
    pro2, best_js2 = get_answer(answer2)
    print('-' * 50, '4:6', '-' * 50)

    # 6:4
    size_A = 30000
    size_B = 20000
    train_A, train_B, train_AB, xy = get_train_Datas(path_A, path_B, size_train, size_A, size_B)
    answer3 = get_pro_JS(train_A, train_B, train_AB, size, xy)
    pro3, best_js3 = get_answer(answer3)
    print('-' * 50, '6:4', '-' * 50)

    # 8:2
    size_A = 30000
    size_B = 7500
    train_A, train_B, train_AB, xy = get_train_Datas(path_A, path_B, size_train, size_A, size_B)
    answer4 = get_pro_JS(train_A, train_B, train_AB, size, xy)
    pro4, best_js4 = get_answer(answer4)
    print('-' * 50, '8:2', '-' * 50)

    w_a = np.linspace(0, 1, 10001)
    y1 = answer1[:, 1]
    y2 = answer2[:, 1]
    y3 = answer3[:, 1]
    y4 = answer4[:, 1]

    df_pro_JS = pd.DataFrame(np.vstack((w_a, y1, y2, y3, y4)).T)   # 数组->DataFrame
    df_pro_JS.columns = ['weight', 'rate28', 'rate46', 'rate64', 'rate82']   # 添加列名
    df_pro_JS.to_csv('df_pro_JS.csv', index=False)   # 写入.csv文件

    plt.plot(w_a, y1, linestyle='dotted', color='b', label='real2:8')
    plt.plot(w_a, y2, linestyle='dashed', color='g', label='real4:6')
    plt.plot(w_a, y3, linestyle='dashdot', color='c', label='real6:4')
    plt.plot(w_a, y4, linestyle='solid', color='y', label='real8:2')
    plt.plot([pro1, pro2, pro3, pro4], [best_js1, best_js2, best_js3, best_js4], 'r*', label='estimation')

    plt.legend()
    plt.savefig('train_plot.png')
    plt.show()


def main_test():
    path_A = '三组分子数据/3S3FLP--37898 events.xlsx'
    path_B = '三组分子数据/6S3FLP--47692 events.xlsx'
    path_AB = '三组分子数据/2. add 6S3FLP--containing 3S3FLP + 6S3FLP------11308 events.xlsx'
    size = 20  # 柱子的个数

    df_A, df_B, df_AB, xy = get_test_Datas(path_A, path_B, path_AB)
    answer = get_pro_JS(df_A, df_B, df_AB, size, xy)
    pro, best_js = get_answer(answer)
    w_a = np.linspace(0, 1, 10001)
    y = answer[:, 1]
    df_pro_JS = pd.DataFrame(np.vstack((w_a, y)).T)  # 数组->DataFrame
    df_pro_JS.columns = ['weight', 'JS']  # 添加列名
    df_pro_JS.to_csv('test_pro_JS4_6.csv', index=False)  # 写入.csv文件

    plt.plot(w_a, y, linestyle='dashed', color='b', label='JS divergence')
    plt.plot([pro], [best_js], 'r*', label='estimation')
    plt.legend()
    plt.savefig('test_plot4_6.png')
    plt.show()

# 训练图
main_draw()

# 测试图
main_test()







