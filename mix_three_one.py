import pandas as pd
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

def get_pro_JS_AandB(df_A, df_B, df_AB, size, xy):
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

def get_pro_JS_ABandC(df_AB, df_C, df_ABC, size, xy):
    # 格子拉直
    print('正在计算AB的经验分布...')
    Z1 = find_all(df_AB, xy, size)
    print('正在计算C的经验分布...')
    Z2 = find_all(df_C, xy, size)
    z1 = Z1.ravel()/np.sum(Z1)
    z2 = Z2.ravel()/np.sum(Z2)
    print('正在计算AB和C混合的经验分布...')
    C = find_all(df_ABC, xy, size)   #混合样本
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


def get_test_Datas(path_A, path_B, path_C, path_AB, path_ABC):
    # 数据读入
    df_A = pd.read_excel(path_A)    # 分子A的数据
    df_B = pd.read_excel(path_B)    # 分子B的数据
    df_AB = pd.read_excel(path_AB)
    df_C = pd.read_excel(path_C)
    df_ABC = pd.read_excel(path_ABC)
    # 第2列数据取对数
    df_A['logY'] = np.log10(df_A.iloc[:,1])
    df_B['logY'] = np.log10(df_B.iloc[:,1])
    df_AB['logY'] = np.log10(df_AB.iloc[:, 1])
    df_C['logY'] = np.log10(df_C.iloc[:, 1])
    df_ABC['logY'] = np.log10(df_ABC.iloc[:,1])
    # 划定格子范围,因闭区间需要,左右放宽1e-7
    # x的最小,x的最大,y的最小,y的最大
    xy = [min(np.min(df_A, axis = 0)[0], np.min(df_B, axis = 0)[0], np.min(df_C, axis = 0)[0])-1e-7,
          max(np.max(df_A, axis = 0)[0], np.max(df_B, axis = 0)[0], np.max(df_C, axis = 0)[0])+1e-7,
          min(np.min(df_A, axis = 0)[2], np.min(df_B, axis = 0)[2], np.min(df_C, axis = 0)[2])-1e-7,
          max(np.max(df_A, axis = 0)[2], np.max(df_B, axis = 0)[2], np.max(df_C, axis = 0)[2])+1e-7]
    return df_A, df_B, df_C, df_AB, df_ABC, xy


def get_answer_AandB(ans, name1, name2):
    res = ans[ans[:,1] == np.min(ans[:,1])]
    pro = res[0, 0]
    best_js = res[0, 1]
    print('%s的权重为[%.4f],%s的权重为[%.4f]时,对应JS散度为[%.6f]'%(name1, pro, name2, 1-pro, best_js))
    return pro, best_js


def main_test():
    path_A = '三组分子数据/3S3FLP--37898 events.xlsx'
    path_B = '三组分子数据/6S3FLP--47692 events.xlsx'
    path_AB = '三组分子数据/2. add 6S3FLP--containing 3S3FLP + 6S3FLP------11308 events.xlsx'
    path_C = '三组分子数据/6S2FLP--43042 events.xlsx'
    path_ABC = '三组分子数据/3. add 6S2FLP--containing 3S3FLP + 6S3FLP +6S2FLP---------20520 events.xlsx'
    size = 20  # 柱子的个数

    df_A, df_B, df_C, df_AB, df_ABC, xy = get_test_Datas(path_A, path_B, path_C, path_AB, path_ABC)
    ans1 = get_pro_JS_AandB(df_A, df_B, df_AB, size, xy)
    pro_A_B, best_js = get_answer_AandB(ans1, 'A', 'B')
    w_a = np.linspace(0, 1, 10001)
    y = ans1[:, 1]
    df_pro_JS = pd.DataFrame(np.vstack((w_a, y)).T)  # 数组->DataFrame
    df_pro_JS.columns = ['weight', 'JS']  # 添加列名
    df_pro_JS.to_csv('test_pro_JSA_B.csv', index=False)  # 写入.csv文件

    plt.figure(1)
    plt.plot(w_a, y, linestyle='dashed', color='b', label='JS divergence')
    plt.plot([pro_A_B], [best_js], 'r*', label='estimation')
    plt.legend()
    plt.title('mix A and B: weight of A')
    plt.savefig('test_plotA_B.png')
    plt.show()

    ans2 = get_pro_JS_ABandC(df_AB, df_C, df_ABC, size, xy)
    pro_AB_C, best_js = get_answer_AandB(ans2, 'AB', 'C')
    y = ans2[:, 1]
    df_pro_JS = pd.DataFrame(np.vstack((w_a, y)).T)  # 数组->DataFrame
    df_pro_JS.columns = ['weight', 'JS']  # 添加列名
    df_pro_JS.to_csv('test_pro_JSAB_C.csv', index=False)  # 写入.csv文件

    plt.figure(2)
    plt.plot(w_a, y, linestyle='dashed', color='b', label='JS divergence')
    plt.plot([pro_AB_C], [best_js], 'r*', label='estimation')
    plt.legend()
    plt.title('mix AB and C: weight of AB')
    plt.savefig('test_plotAB_C.png')
    plt.show()

    prob_C = 1-pro_AB_C
    prob_A = pro_AB_C * pro_A_B
    prob_B = pro_AB_C * (1-pro_A_B)
    print('最终结果：A占比[%.4f],B占比[%.4f],C占比[%.4f]'%(prob_A, prob_B, prob_C))

main_test()






