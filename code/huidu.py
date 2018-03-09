import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
def huidu(data,x0):
    X0=data[x0]
    #对于空值未处理
    data.loc[50] = data.apply(lambda x: x.mean())
    for i in data.columns:
        start=data.ix[50,i]
        for j in data.index:
            if start!=0:
                data.ix[j, i] = (1.0 * data.ix[j, i] / start)
            else:
                pass



    M=0  #max
    N=100000  #min
    for i in data.columns.drop(x0):
        for j in data[i].index:
            if np.abs(data.ix[j,i]-data.ix[j,x0])>M:
                M=np.abs(data.ix[j,i]-data.ix[j,x0])
            if np.abs(data.ix[j,i]-data.ix[j,x0])<N:
                N=np.abs(data.ix[j,i]-data.ix[j,x0])


    for i in data.columns.drop(x0):


        for j in data[i].index:
            # print(data[i].index)
            data.ix[j,i]=(N+0.5*M)/(np.abs(data.ix[j,i]-data.ix[j,x0])+0.5*M)

    data.drop(50,inplace=True)
    data.loc['avarage'] = data.apply(lambda x: x.mean())

    return data

def rlt(data,name):
    fig=plt.figure(name)
    corr=data.corr()

    xlabels=corr.index
    ylabels=corr.index
    ax1 = fig.add_subplot(1, 1, 1)
    sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1, annot_kws={'size': 8, 'weight': 'bold', 'color': 'blue'})
    ax1.set_xticklabels(xlabels, rotation=0, fontsize=5)
    ax1.set_yticklabels(ylabels, rotation=0, fontsize=5)


data=pd.read_csv("D:\meisai\AZRT1.csv")
data=data.ix[0:,4:]
rlt(data,"AZRT")
data=data.dropna()
data=huidu(data,"R_T")
data.to_csv("D:\meisai\huidu\AZRT1.csv")

data=pd.read_csv("D:\meisai\AZRNT1.csv")
data=data.ix[0:,4:]
rlt(data,"AZRNT")
data=data.dropna()
data=huidu(data,"R_N_T")
data.to_csv("D:\meisai\huidu\AZRNT1.csv")


data=pd.read_csv("D:\meisai\AZTETCB1.csv")
data=data.ix[0:,2:]
data=data.dropna()
rlt(data,"AZTETCB")
data=huidu(data,"TETCB_Data")
data.to_csv("D:\meisai\huidu\AZTETCB1.csv")

data=pd.read_csv("D:\meisai\CARNT1.csv")
data=data.ix[0:,4:]
data=data.dropna()
rlt(data,"CARNT")
data=huidu(data,"R_N_T")
data.to_csv("D:\meisai\huidu\CARNT1.csv")

data=pd.read_csv("D:\meisai\CATETCB1.csv")
data=data.ix[0:,2:]
data=data.dropna()
rlt(data,"CATETCB")
data=huidu(data,"TETCB_Data")
data.to_csv("D:\meisai\huidu\CATETCB1.csv")

data=pd.read_csv("D:\meisai\CART1.csv")
data=data.ix[0:,4:]
data=data.dropna()
rlt(data,"CART")
data=huidu(data,"R_T")
data.to_csv("D:\meisai\huidu\CART1.csv")


data=pd.read_csv("D:\meisai\jNMRNT1.csv")
data=data.ix[0:,4:]
data=data.dropna()
rlt(data,"NMRNT")
data=huidu(data,"R_N_T")
data.to_csv("D:\meisai\huidu\jNMRNT1.csv")

data=pd.read_csv("D:\meisai\jNMTETCB1.csv")
data=data.ix[0:,2:]
data=data.dropna()
rlt(data,"NMTETCB")
data=huidu(data,"TETCB_Data")
data.to_csv("D:\meisai\huidu\jNMTETCB1.csv")

data=pd.read_csv("D:\meisai\jNMRT1.csv")
data=data.ix[0:,4:]
data=data.dropna()
rlt(data,"NMRT")
data=huidu(data,"R_T")
data.to_csv("D:\meisai\huidu\jNMRT1.csv")

data=pd.read_csv("D:\meisai\TXRNT1.csv")
data=data.ix[0:,4:]
data=data.dropna()
rlt(data,"TXRNT")
data=huidu(data,"R_N_T")
data.to_csv("D:\meisai\huidu\TXRNT1.csv")

data=pd.read_csv("D:\meisai\TXTETCB1.csv")
data=data.ix[0:,2:]
data=data.dropna()
rlt(data,"TXTETCB")
data=huidu(data,"TETCB_Data")
data.to_csv("D:\meisai\huidu\TXTETCB1.csv")

data=pd.read_csv("D:\meisai\TXRT1.csv")
data=data.ix[0:,4:]
data=data.dropna()
rlt(data,"TXRT")
data=huidu(data,"R_T")
data.to_csv("D:\meisai\huidu\TXRT1.csv")

#################################################后77-09##################
data=pd.read_csv("D:\meisai\AZRT.csv")
data=data.ix[0:,2:].drop(['RETCB_Data','TETCB_Data'],1)
# data=data.drop('TETCB_Data',1)
rlt(data,"AZRT")
data=data.dropna()
data=huidu(data,"R_T")
data.to_csv("D:\meisai\huidu\AZRT.csv")

data=pd.read_csv("D:\meisai\AZRNT.csv")
data=data.ix[0:,2:].drop(['RETCB_Data','TETCB_Data'],1)
rlt(data,"AZRNT")
data=data.dropna()
data=huidu(data,"R_N_T")
data.to_csv("D:\meisai\huidu\AZRNT.csv")


data=pd.read_csv("D:\meisai\AZTETCB.csv")
data=data.ix[0:,2:]
data=data.dropna()
rlt(data,"AZTETCB")
data=huidu(data,"TETCB_Data")
data.to_csv("D:\meisai\huidu\AZTETCB.csv")

data=pd.read_csv("D:\meisai\CARNT.csv")
data=data.ix[0:,2:].drop(['RETCB_Data','TETCB_Data'],1)
data=data.dropna()
rlt(data,"CARNT")
data=huidu(data,"R_N_T")
data.to_csv("D:\meisai\huidu\CARNT.csv")

data=pd.read_csv("D:\meisai\CATETCB.csv")
data=data.ix[0:,2:]
data=data.dropna()
rlt(data,"CATETCB")
data=huidu(data,"TETCB_Data")
data.to_csv("D:\meisai\huidu\CATETCB.csv")

data=pd.read_csv("D:\meisai\CART.csv")
data=data.ix[0:,2:].drop(['RETCB_Data','TETCB_Data'],1)
data=data.dropna()
rlt(data,"CART")
data=huidu(data,"R_T")
data.to_csv("D:\meisai\huidu\CART.csv")


data=pd.read_csv("D:\meisai\jNMRNT.csv")
data=data.ix[0:,2:].drop(['RETCB_Data','TETCB_Data'],1)
data=data.dropna()
rlt(data,"NMRNT")
data=huidu(data,"R_N_T")
data.to_csv("D:\meisai\huidu\jNMRNT.csv")

data=pd.read_csv("D:\meisai\jNMTETCB.csv")
data=data.ix[0:,2:]
data=data.dropna()
rlt(data,"NMTETCB")
data=huidu(data,"TETCB_Data")
data.to_csv("D:\meisai\huidu\jNMTETCB.csv")

data=pd.read_csv("D:\meisai\jNMRT.csv")
data=data.ix[0:,2:].drop(['RETCB_Data','TETCB_Data'],1)
data=data.dropna()
rlt(data,"NMRT")
data=huidu(data,"R_T")
data.to_csv("D:\meisai\huidu\jNMRT.csv")

data=pd.read_csv("D:\meisai\TXRNT.csv")
data=data.ix[0:,2:].drop(['RETCB_Data','TETCB_Data'],1)
data=data.dropna()
rlt(data,"TXRNT")
data=huidu(data,"R_N_T")
data.to_csv("D:\meisai\huidu\TXRNT.csv")

data=pd.read_csv("D:\meisai\TXTETCB.csv")
data=data.ix[0:,2:]
data=data.dropna()
rlt(data,"TXTETCB")
data=huidu(data,"TETCB_Data")
data.to_csv("D:\meisai\huidu\TXTETCB.csv")

data=pd.read_csv("D:\meisai\TXRT.csv")
data=data.ix[0:,2:].drop(['RETCB_Data','TETCB_Data'],1)
data=data.dropna()
rlt(data,"TXRT")
data=huidu(data,"R_T")
data.to_csv("D:\meisai\huidu\TXRT.csv")

plt.show()
# path=["D:\meisai\huidu\TXRNT.csv","D:\meisai\huidu\TXTETCB.csv","D:\meisai\huidu\TXRT.csv",
#       ]

###后40年的预测对比 GM-NN
# data1 = newpdall.loc[range(2010, 2051)].copy()
# data_train = data1
# data_mean = data_train.mean()
# data_std = data_train.std()
# data_train = (data_train - data_mean) / data_std
#
#
# x_train = data_train[feature].as_matrix()
#
# y_train = data_train[y].as_matrix()
# model.fit(x_train, y_train, epochs=10000, batch_size=16)
# model.save_weights(state + y[0] + '2010_2050.model')
# x =data_train[feature].as_matrix()
#
#
# t = model.predict(x)
# h = []
# for i in range(33):
#     print('t', t[i][0])
#     print('std', data_std.ix[0, y])
#     print('mean', data_mean.ix[0, y])
#     h.append((t[i][0] * data_std[y] + data_mean[y]))
# print(h)
# data1['2010-2050_prediction'] = h
#
# data1.to_csv('NN_prediction_' + state + y[0] + '20-50.csv')
#
# fig = plt.figure('NN_prediction_' + state + y[0])
# plt.plot(range(len(data1['y_prediction'].index)), data1['y_prediction'].values, 'b', label="NN_predict")
# plt.plot(range(len(data1[y].index)), data1[y].values, 'r', label=(y[0] + 'GM_values'))
# plt.legend(loc="upper right")  # 显示图中的标签






