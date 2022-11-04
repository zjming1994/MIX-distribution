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


def get_pro_JS(df_A, df_B, df_C, df_ABC, size, xy, w_x, w_y):
    # 格子拉直
    print('正在计算A的经验分布...')
    Z1 = find_all(df_A, xy, size)
    print('正在计算B的经验分布...')
    Z2 = find_all(df_B, xy, size)
    print('正在计算C的经验分布...')
    Z3 = find_all(df_C, xy, size)

    z1 = Z1.ravel()/np.sum(Z1)
    z2 = Z2.ravel()/np.sum(Z2)
    z3 = Z3.ravel() / np.sum(Z3)
    print('正在计算A和B和C混合的经验分布...')
    D = find_all(df_ABC, xy, size)   #混合样本
    print('经验分布计算结束！')
    d = D.ravel()/np.sum(D)
    # 网格搜索最优JS散度和混合值,保留4位有效数字
    ans = np.full((101, 101), np.nan)

    print('正在进行网格搜索...')
    for i in range(101):
        for j in range(101):
            if w_x[i] + w_y[j] <= 1:
                k = 1-w_x[i]-w_y[j]
                js = JS_divergence(w_x[i]*z1 + w_y[j]*z2 + k*z3, d)
                ans[i,j] = js
    print('网格搜索结束！')
    return ans



def get_test_Datas(path_A, path_B, path_C, path_ABC):
    # 数据读入
    df_A = pd.read_excel(path_A)    # 分子A的数据
    df_B = pd.read_excel(path_B)    # 分子B的数据
    df_C = pd.read_excel(path_C)  # 分子B的数据
    df_ABC = pd.read_excel(path_ABC)    # 混合分子的数据（真实）
    df_ABC = df_ABC.astype('float')
    # 第2列数据取对数
    df_A['logY'] = np.log10(df_A.iloc[:,1])
    df_B['logY'] = np.log10(df_B.iloc[:,1])
    df_C['logY'] = np.log10(df_C.iloc[:, 1])
    df_ABC['logY'] = np.log10(df_ABC.iloc[:,1])
    # 划定格子范围,因闭区间需要,左右放宽1e-7
    # x的最小,x的最大,y的最小,y的最大
    xy = [min(np.min(df_A, axis = 0)[0], np.min(df_B, axis = 0)[0], np.min(df_C, axis = 0)[0])-1e-7,
          max(np.max(df_A, axis = 0)[0], np.max(df_B, axis = 0)[0], np.max(df_C, axis = 0)[0])+1e-7,
          min(np.min(df_A, axis = 0)[2], np.min(df_B, axis = 0)[2], np.min(df_C, axis = 0)[2])-1e-7,
          max(np.max(df_A, axis = 0)[2], np.max(df_B, axis = 0)[2], np.max(df_C, axis = 0)[2])+1e-7]
    return df_A, df_B, df_C, df_ABC, xy


def get_answer(ans):
    nan_pos = np.isnan(ans)
    ans[nan_pos] = -1
    inf_pos = np.isinf(ans)
    ans[inf_pos] = -1
    ans[ans == -1] = np.max(ans)
    index = np.argmin(ans)
    index_i, index_j = index // 101, index % 101
    JS = np.min(ans)
    return index_i, index_j, JS, ans

def draw(w_x, w_y, ans, best_x, best_y, best_JS):
    fig = plt.figure()  # 定义新的三维坐标轴
    ax3 = plt.axes(projection='3d')
    Y, X = np.meshgrid(w_x, w_y)
    ax3.plot_surface(X, Y, ans, cmap='rainbow')
    ax3.contour(X, Y, ans, offset=best_JS, cmap='rainbow')  # 等高线图，要设置offset，为Z的最小值
    ax3.set_xlabel('A')
    ax3.set_ylabel('B')
    ax3.set_zlabel('JS')
    ax3.scatter3D(best_x, best_y, best_JS, s=100, color='r', marker='o')  # 绘制散点图
    plt.show()

def main_test():
    path_A = '三组分子数据/3S3FLP--37898 events.xlsx'
    path_B = '三组分子数据/6S3FLP--47692 events.xlsx'
    path_C = '三组分子数据/6S2FLP--43042 events.xlsx'
    # path_ABC = 'E:/MIX_distribution/三组分子数据/1. add 3S3FLP--only 3SFLP----9064 events.xlsx'
    # path_ABC = 'E:/MIX_distribution/三组分子数据/2. add 6S3FLP--containing 3S3FLP + 6S3FLP------11308 events.xlsx'
    path_ABC = '三组分子数据/3. add 6S2FLP--containing 3S3FLP + 6S3FLP +6S2FLP---------20520 events.xlsx'
    size = 20  # 柱子的个数

    df_A, df_B, df_C, df_ABC, xy = get_test_Datas(path_A, path_B, path_C, path_ABC)

    print('正在进行第一轮搜索...')
    w_x = np.linspace(0,1,101)
    w_y = np.linspace(0,1,101)
    ans1 = get_pro_JS(df_A, df_B, df_C, df_ABC, size, xy, w_x, w_y)
    index_i, index_j, JS1, ans1 = get_answer(ans1)
    df_ans1 = pd.DataFrame(ans1)  # 数组->DataFrame
    df_ans1.to_csv('df_ans1.csv', index=False)  # 写入.csv文件
    print('第一轮搜索结果:A比例[%.2f]:B比例[%.2f]:C比例[%.2f],此时JS散度最小,为[%.6f]'
              %(w_x[index_i], w_y[index_j], 1-w_x[index_i]-w_y[index_j], JS1))


    print('正在进行第二轮搜索...')
    if index_i == 0:
        w_a = np.linspace(0, 0.01, 101)
    elif index_i == 1:
        w_a = np.linspace(0.99, 1, 101)
    else:
        w_a = np.linspace(w_x[index_i] - 0.005, w_x[index_i] + 0.005, 101)

    if index_j == 0:
        w_b = np.linspace(0, 0.01, 101)
    elif index_j == 1:
        w_b = np.linspace(0.99, 1, 101)
    else:
        w_b = np.linspace(w_y[index_j] - 0.005, w_y[index_j] + 0.005, 101)

    ans2 = get_pro_JS(df_A, df_B, df_C, df_ABC, size, xy, w_a, w_b)
    # print(w_a, w_b)
    index_ii, index_jj, JS2, ans2 = get_answer(ans2)
    df_ans2 = pd.DataFrame(ans2)  # 数组->DataFrame
    df_ans2.to_csv('df_ans2.csv', index=False)  # 写入.csv文件
    print('第二轮搜索结果:A比例[%.4f]:B比例[%.4f]:C比例[%.4f],此时JS散度最小,为[%.6f]'
              %(w_a[index_ii], w_b[index_jj], 1-w_a[index_ii]-w_b[index_jj], JS2))
    # draw(w_a, w_b, ans2, w_a[index_ii], w_b[index_jj], JS2)
    draw(w_x, w_y, ans1, w_x[index_i], w_y[index_j], JS1)

main_test()











