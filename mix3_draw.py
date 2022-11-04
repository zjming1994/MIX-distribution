import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt

path_A = '三组分子数据/3S3FLP--37898 events.xlsx'
path_B = '三组分子数据/6S3FLP--47692 events.xlsx'
path_C = '三组分子数据/6S2FLP--43042 events.xlsx'

df_A = pd.read_excel(path_A)        # 分子A的数据
df_B = pd.read_excel(path_B)        # 分子B的数据
df_C = pd.read_excel(path_C)        # 分子C的数据

# 第2列数据取对数
df_A['logY'] = np.log10(df_A.iloc[:,1])
df_B['logY'] = np.log10(df_B.iloc[:,1])
df_C['logY'] = np.log10(df_C.iloc[:, 1])

size_train = 3000
index = np.random.choice(a=df_A.shape[0], size=size_train, replace=False)

df_A = df_A.iloc[index,:]
df_B = df_B.iloc[index,:]
df_C = df_C.iloc[index,:]

df_ABC = np.vstack((df_A.iloc[:1000,:], df_B.iloc[:1000,:], df_C.iloc[:1000,:]))
df_ABC = pd.DataFrame(df_ABC)

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


xy = [0.1, 1, -1, 1.3]
size = 30
w_x = np.linspace(xy[0], xy[1], size)
w_y = np.linspace(xy[2], xy[3], size)
Z_A = find_all(df_A, xy, size)
Z_B = find_all(df_B, xy, size)
Z_C = find_all(df_C, xy, size)
Z_ABC = find_all(df_ABC, xy, size)


def draw(w_x, w_y, ans, c):
    fig = plt.figure()  # 定义新的三维坐标轴
    ax3 = plt.axes(projection='3d')
    X, Y = np.meshgrid(w_x, w_y)
    ax3.plot_surface(X, Y, ans/3000, cmap=c)
    plt.show()

draw(w_x, w_y, Z_A, 'Reds')
draw(w_x, w_y, Z_B, 'Blues')
draw(w_x, w_y, Z_C, 'Greens')

draw(w_x, w_y, Z_ABC, 'binary')




