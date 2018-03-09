import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback
from keras.datasets import mnist
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.utils import np_utils
from phased_lstm_keras.PhasedLSTM import PhasedLSTM as PLSTM
import pandas as pd
rnn_unit=10
#####变化
input_size=5###len(feather)
output_size=1
lr=0.0006
mopath='S:/pycharm/Cmin/lstmmodel/'
CHECKPOINT_PATH='lstmmodel/lsmtmodel.meta'

#获取训练集
def get_train_data(data,feather,Y11,batch_size=6,time_step=2,train_begin=0,train_end=24):
    batch_index=[]
    column=list(data.columns)
    XX=[]
    YY=column.index(Y11)
    for da in feather:
        XX.append(column.index(da))



    data_train=data[train_begin:train_end].as_matrix()
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,XX]
       y=normalized_train_data[i:i+time_step,YY,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y



#获取测试集
def get_test_data(data,feather,Y11,time_step=2,test_begin=24):
    column = list(data.columns)
    XX = []
    YY = column.index(Y11)
    for da in feather:
        XX.append(column.index(da))

    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample
    test_x,test_y=[],[]
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,XX]
       y=normalized_test_data[i*time_step:(i+1)*time_step,YY]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,XX]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,YY]).tolist())
    return mean,std,test_x,test_y




class AccHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('acc'))


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

data = pd.read_csv('D:\meisai\CARNT.csv')
data = data.ix[0:, 2:]
data = data.dropna()
T = data.columns.drop(['R_N_T', 'TETCB_Data', 'RETCB_Data'], 1)
T = T[0:5]

batch_index,train_x,train_y=get_train_data(data,T,'R_N_T')
mean,std,test_x,test_y=get_test_data(data,T,'R_N_T')

model_PLSTM = Sequential()
model_PLSTM.add(PLSTM(32, input_shape=(28 * 28, 1), implementation=2))
model_PLSTM.add(Dense(10, activation='softmax'))
model_PLSTM.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                        metrics=['accuracy'])
model_PLSTM.summary()
acc_PLSTM = AccHistory()
loss_PLSTM = LossHistory()
model_PLSTM.fit(train_x, train_y, epochs=10000, batch_size=12,
                    callbacks=[acc_PLSTM, loss_PLSTM])
score_PLSTM = model_PLSTM.evaluate(test_x, test_y, verbose=0)
print(score_PLSTM)

