import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers.core import  Dense,Activation
from keras.layers import Activation, Convolution2D, Dropout
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
class GM11:


    def __init__(self, sequence, phio=0.5):
        self.sequence = np.array(sequence)
        self.a_hat = self.__getParameters()
        self.phio = phio

    def __getParameters(self):

        X1 = np.cumsum(self.sequence).transpose()
        X1_temp = (X1[:-1] + X1[1:]) / 2
        B = np.column_stack((-X1_temp, np.ones_like(X1_temp)))
        Y = self.sequence[1:]
        a_hat = np.dot(np.linalg.inv(np.dot(B.T, B)), np.dot(B.T, Y))
        return a_hat

    def gm11_predict_test(self, method=None):
        '''
        Model test of GM(1,1)
        First method: Residual test
        Second method: Correlation degree test
        Third method: Posterior difference test
        '''
        X0_hat = self.gm11Predict(len(self.sequence))
        delt_0 = abs(self.sequence - X0_hat)

        # Residual test
        if method == 'Residual_test':
            Fi = delt_0 / self.sequence
            return Fi

        # Correlation degree test
        elif method == 'Correlation_degree_test':
            yita = (min(delt_0) + self.phio * max(delt_0)) / (
                delt_0 + self.phio * max(delt_0))
            R = np.mean(yita)
            return R

        # Posterior difference test
        elif method == 'Posterior_difference_test':
            C = np.std(delt_0) / np.std(self.sequence)
            P = np.sum((delt_0 - np.mean(delt_0)) < 0.674 *
                       np.std(self.sequence)) / len(delt_0)
            if P > 0.95 and C < 0.35:
                print('Model good')
            elif P > 0.80 and C < 0.50:
                print('Model standard')
            elif P > 0.70 and C < 0.65:
                print('Barely qualified')
            else:
                print('Model bad')

    def gm11Predict(self, n):

        X_hat = [(self.sequence[0] - (self.a_hat[1] / self.a_hat[0])) *
                 np.exp(-self.a_hat[0] * k) + self.a_hat[1] / self.a_hat[0]
                  for k in range(n)]

        X0_hat = np.diff(np.array(X_hat))
        X0_hat = np.insert(X0_hat, 0, X_hat[0])
        return X0_hat


#####机器学习模型############
# feature是特征数组，y是目标变量数组
def ml_model(data,state,feature,y):

    # print('data\n',data)
    ###peason 相关系数模型
    corr=data.corr(method='pearson')
    fig=plt.figure(state+y[0])
    fig.set_size_inches(11,8)
    xlabels = corr.index
    ylabels = corr.index
    ax1 = fig.add_subplot(1, 1, 1)
    sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1, annot_kws={'size': 8, 'weight': 'bold', 'color': 'blue'})
    ax1.set_xticklabels(xlabels, rotation=0, fontsize=6)
    ax1.set_yticklabels(ylabels, rotation=0, fontsize=10)
    fig.savefig(state+y[0]+'.png')
    #特征选择待定
    #GM
    newpd=pd.DataFrame(columns=data.columns)
    newpdall=pd.DataFrame(columns=data.columns)

    for i in data.columns.drop(y[0]):
        result=GM11(data.ix[1977:,i])
        h=result.gm11Predict(74)
        newpd[i]=pd.Series(h)
    newpd.index=range(1977,2051)
    for i in data.columns:
        result=GM11(data.ix[1977:,i])
        h=result.gm11Predict(74)
        newpdall[i]=pd.Series(h)
    newpdall.index=range(1977,2051)
    yuce=pd.DataFrame(columns=['Date',state+y[0]])
    yuce.loc[0]=[2025,newpdall.ix[2025,y[0]]]
    yuce.loc[1]=[2050,newpdall.ix[2050,y[0]]]
    yuce.to_csv(state+'  '+y[0]+"预测表.csv")



    GMdata=data.append(newpd.loc[2010:2050])

    GMdata.to_csv(state+y[0]+'GM_Prediction.csv')

    data1=GMdata.loc[range(1977,2010)].copy()
    data_train=data1
    data_mean=data_train.mean()
    data_std=data_train.std()
    data_train=(data_train-data_mean)/data_std

    x_train=data_train[feature].as_matrix()

    y_train=data_train[y].as_matrix()

    model=Sequential()
    model.add(Dense(2*len(feature),input_shape=(len(feature),)))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Dropout(.3))

    model.compile(loss='mean_squared_error',optimizer='adam')
    model.summary()

    model.fit(x_train,y_train,epochs=10000,batch_size=16)

    model.save_weights(state+y[0]+'.model')

    x=((data1[feature]-data_mean[feature])/data_std[feature]).as_matrix()


    t=model.predict(x)
    h=[]
    MSE=0
    for i in range(33):
        # print('t',t[i][0])
        # print('std',data_std.ix[0,y])
        # print('mean',data_mean.ix[0,y])
        h.append((t[i][0]*data_std[y]+data_mean[y]))
        MSE+=((h[i]-data1.ix[i+1977,y])/(data1.ix[i+1977,y]))**2
    MSE=1.0*MSE/33

    data1[y[0]+'  prediction  values']=h

    GMdata.to_csv('NN_prediction_'+state+y[0]+'.csv')

    fig=plt.figure('prediction  '+state+'  '+y[0])
    plt.plot(range(len(data1[y[0]+'  prediction  values'].index)),data1[y[0]+'  prediction  values'].values,'b',label=(y[0]+"prediction values"))
    plt.plot(range(len(data1[y].index)), data1[y].values, 'r', label=(y[0]+'  true   values'))
    print('MSE',MSE)
    yuce.loc['MSE'] =[None,MSE]
    plt.legend(loc="upper right")  # 显示图中的标签
    plt.savefig('prediction  '+state+'  '+y[0]+'.png')
    return yuce


data=pd.read_csv('D:\meisai\AZRNT.csv')
data.index=data['Year']
data=data.ix[0:,1:]
data=data.drop('Year',1)
data=data.dropna()
# print(data.columns.drop(['R_T','TETCB_Data','RETCB_Data'],1))
yuce=ml_model(data,'AZ',data.columns.drop(['R_N_T','TETCB_Data','RETCB_Data'],1),['R_N_T'])

data=pd.read_csv('D:\meisai\AZRT.csv')
data.index=data['Year']
data=data.ix[0:,1:]
data=data.drop('Year',1)
data=data.dropna()
# print(data.columns.drop(['R_T','TETCB_Data','RETCB_Data'],1))
yuce1=ml_model(data,'AZ',data.columns.drop(['R_T','TETCB_Data','RETCB_Data'],1),['R_T'])

result=pd.merge(yuce,yuce1,how='outer')

data=pd.read_csv('D:\meisai\AZTETCB.csv')
data.index=data['Year']
data=data.ix[0:,1:]
data=data.drop('Year',1)
data=data.dropna()
# print(data.columns.drop(['R_T','TETCB_Data','RETCB_Data'],1))
yuce1=ml_model(data,'AZ',data.columns.drop(['TETCB_Data'],1),['TETCB_Data'])

result=pd.merge(result,yuce1,how='outer')
#
# ########################CA
data=pd.read_csv('D:\meisai\CARNT.csv')
data.index=data['Year']
data=data.ix[0:,1:]
data=data.drop('Year',1)
data=data.dropna()
# print(data.columns.drop(['R_T','TETCB_Data','RETCB_Data'],1))
yuce1=ml_model(data,'CA',data.columns.drop(['R_N_T','TETCB_Data','RETCB_Data'],1),['R_N_T'])
result=pd.merge(result,yuce1,how='outer')
#
#
#
data=pd.read_csv('D:\meisai\CART.csv')
data.index=data['Year']
data=data.ix[0:,1:]
data=data.drop('Year',1)
data=data.dropna()
# print(data.columns.drop(['R_T','TETCB_Data','RETCB_Data'],1))
yuce1=ml_model(data,'CA',data.columns.drop(['R_T','TETCB_Data','RETCB_Data'],1),['R_T'])
result=pd.merge(result,yuce1,how='outer')
#
data=pd.read_csv('D:\meisai\CATETCB.csv')
data.index=data['Year']
data=data.ix[0:,1:]
data=data.drop('Year',1)
data=data.dropna()
# print(data.columns.drop(['R_T','TETCB_Data','RETCB_Data'],1))
yuce1=ml_model(data,'CA',data.columns.drop(['TETCB_Data'],1),['TETCB_Data'])
result=pd.merge(result,yuce1,how='outer')
#
# ###################NM
data=pd.read_csv('D:\meisai\jNMRNT.csv')
data.index=data['Year']
data=data.ix[0:,1:]
data=data.drop('Year',1)
data=data.dropna()
# print(data.columns.drop(['R_T','TETCB_Data','RETCB_Data'],1))
yuce1=ml_model(data.drop(['NUETB_Data','NUETD_Data'],1),'NM',data.columns.drop(['R_N_T','TETCB_Data','RETCB_Data','NUETB_Data','NUETD_Data'],1),['R_N_T'])
result=pd.merge(result,yuce1,how='outer')

#
#
data=pd.read_csv('D:\meisai\jNMRT.csv')
data.index=data['Year']
data=data.ix[0:,1:]
data=data.drop('Year',1)
data=data.dropna()
# print(data.columns.drop(['R_T','TETCB_Data','RETCB_Data'],1))
yuce1=ml_model(data,'NM',data.columns.drop(['R_T','TETCB_Data','RETCB_Data'],1),['R_T'])
result=pd.merge(result,yuce1,how='outer')
#
data=pd.read_csv('D:\meisai\jNMTETCB.csv')
data.index=data['Year']
data=data.ix[0:,1:]
data=data.drop('Year',1)
data=data.dropna()
# print(data.columns.drop(['R_T','TETCB_Data','RETCB_Data'],1))
yuce1=ml_model(data,'NM',data.columns.drop(['TETCB_Data'],1),['TETCB_Data'])
result=pd.merge(result,yuce1,how='outer')
#
# ###############TX
data=pd.read_csv('D:\meisai\TXRNT.csv')
data.index=data['Year']
data=data.ix[0:,1:]
data=data.drop('Year',1)
data=data.dropna()
# print(data.columns.drop(['R_T','TETCB_Data','RETCB_Data'],1))
yuce1=ml_model(data,'TX',data.columns.drop(['R_N_T','TETCB_Data','RETCB_Data'],1),['R_N_T'])
result=pd.merge(result,yuce1,how='outer')
#
#
#
data=pd.read_csv('D:\meisai\TXRT.csv')
data.index=data['Year']
data=data.ix[0:,1:]
data=data.drop('Year',1)
data=data.dropna()
# print(data.columns.drop(['R_T','TETCB_Data','RETCB_Data'],1))
yuce1=ml_model(data,'TX',data.columns.drop(['R_T','TETCB_Data','RETCB_Data'],1),['R_T'])
result=pd.merge(result,yuce1,how='outer')
#
data=pd.read_csv('D:\meisai\TXTETCB.csv')
data.index=data['Year']
data=data.ix[0:,1:]
data=data.drop('Year',1)
data=data.dropna()
# print(data.columns.drop(['R_T','TETCB_Data','RETCB_Data'],1))
yuce1=ml_model(data,'TX',data.columns.drop(['TETCB_Data'],1),['TETCB_Data'])
result=pd.merge(result,yuce1,how='outer')

result.to_csv('所有州三类目标变量预测表.csv')

# plt.show()


