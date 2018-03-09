import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def find_RETCB(data,state,year):
    return data[["Data"]][(data.MSN=="RETCB") & (data.StateCode==state) & (data.Year==year)]
def find_REPRB(data,state,year):
    return data[["Data"]][(data.MSN=="REPRB") & (data.StateCode==state) & (data.Year==year)]
def find_PETCD(data,state,year):
    return data[["Data"]][(data.MSN=="PETCD") & (data.StateCode==state) & (data.Year==year)]
def find_PETCV(data,state,year):
    return data[["Data"]][(data.MSN=="PETCV") &(data.StateCode==state) & (data.Year==year)]
def merge_data(data,state):
    i = data.loc[(data['MSN'] == 'RETCB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "RETCB_Data"})
    j = data.loc[(data['MSN'] == 'REPRB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "REPRB_Data"})
    z=pd.merge(i,j,how='outer')
    j = data.loc[(data['MSN'] == 'PETCD') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "PETCD_Data"})
    z=pd.merge(z,j,how="outer")
    j = data.loc[(data['MSN'] == 'PETCV') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "PETCV_Data"})
    z = pd.merge(j, z, how='outer')
    j = data.loc[(data['MSN'] == 'NUETB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "NUETB_Data"})
    z = pd.merge(z, j, how='outer')
    j = data.loc[(data['MSN'] == 'TETCB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TETCB_Data"})
    z = pd.merge(z, j, how='outer')
    # featherdata=pd.read_excel("D:\meisai\cdata.xlsx",sheetname='msncodes')
    # feather=featherdata.ix[0:,[0]]
    # for i in range(len(feather.index)):
    #     # print(feather.iat[i,0])
    #     if((feather.iat[i,0]!="RETCB") & (feather.iat[i,0]!="REPRB") & (feather.iat[i,0]!="PETCD") & (feather.iat[i,0]!="PETCV")):
    #         j = data.loc[(data['MSN'] == feather.iat[i,0]) & (data['StateCode'] == state), ["Year", "Data"]].rename(
    #             columns={"Data": feather.iat[i,0]+"_Data"})
    #         z = pd.merge(z, j, how='outer')




    PETCB_Data=[]
    R_T=[]
    P_P_R=[]
    R_N=[]
    R_N_T=[]
    for i in range(len(z.index)):
        PETCB_Data.append(1000*z.iat[i,1]/z.iat[i,4])
        P_P_R.append(1000*z.iat[i,1]/z.iat[i,4]-z.iat[i,2])
        R_T.append(z.iat[i,2]/z.iat[i,6])
        R_N.append(z.iat[i,2]+z.iat[i,5])
        R_N_T.append(R_N[i]/z.iat[i,6])
    z['PETCB_Data']=PETCB_Data
    z['R_T']=R_T
    z['P_P_R']=P_P_R
    z["R_N"]=R_N
    z["R_N_T"]=R_N_T

    w=z.loc[40:]
    w=w.append(z.loc[0:39],ignore_index=True)

    return w
data=pd.read_excel("D:\meisai\cdata.xlsx")
state=["AZ","CA","NM","TX"]

AZ_Data=merge_data(data,state[0])
# AZ_Data.to_csv("D:\meisai\AZ_Data_ALL.csv")
AZ_Data=AZ_Data.fillna(0)
# del AZ_Data['Year']
AZ_Data.columns=range(0,611,1)
pca=PCA(n_components=3)
X, y = AZ_Data.iloc[:, 1:].values, AZ_Data.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)
lr = LogisticRegression()
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
print(lr.score(X_test_pca, y_test))
#
#
AZ_pca=pca.fit_transform(AZ_Data)
print(AZ_pca.shape)

CA_Data=merge_data(data,state[1])
CA_Data.to_csv("D:\meisai\CA_Data_ALL.csv")
NM_Data=merge_data(data,state[2])
NM_Data.to_csv("D:\meisai\hNM_Data_ALL.csv")
TX_Data=merge_data(data,state[-1])
TX_Data.to_csv("D:\meisai\TX_Data_ALL.csv")

fig=plt.figure("RETCB")
fig.set(alpha=0.2)
x1 = data[["Year"]][(data.MSN == "RETCB") & (data.StateCode == "AZ")]
y1 = data[["Data"]][(data.MSN == "RETCB") & (data.StateCode == "AZ")]
x2= data[["Year"]][(data.MSN == "RETCB") & (data.StateCode == "CA")]
y2=data[["Data"]][(data.MSN == "RETCB") & (data.StateCode == "CA")]
x3=data[["Year"]][(data.MSN == "RETCB") & (data.StateCode == "NM")]
y3=data[["Data"]][(data.MSN == "RETCB") & (data.StateCode == "NM")]
x4=data[["Year"]][(data.MSN == "RETCB") & (data.StateCode == "TX")]
y4=data[["Data"]][(data.MSN == "RETCB") & (data.StateCode == "TX")]
plt.plot(x1, y1,x2,y2,x3,y3,x4,y4)
plt.xlabel("year")
plt.ylabel("RETCB_Data")
plt.title("line chart for total renewable energy consumption over the year")
plt.xticks(np.arange(1960, 2010, 1), fontsize=5)
plt.legend(("AZ_RETCB","CA_RETCB","NM_RETCB","TX_RETCB"))

fig=plt.figure("Non-renewable energy(P_P_R)")
fig.set(alpha=0.2)
x1 = AZ_Data["Year"]
y1 = AZ_Data["P_P_R"]
x2= x1
y2=CA_Data["P_P_R"]
x3=x1
y3=NM_Data["P_P_R"]
x4=x1
y4=TX_Data["P_P_R"]
plt.plot(x1, y1,x2,y2,x3,y3,x4,y4)
plt.xlabel("year")
plt.ylabel("Non-renewable energy(P_P_R)")
plt.title("line chart for non-renewable energy consumption over the year")
plt.xticks(np.arange(1970, 2010, 1), fontsize=5)
plt.legend(("AZ_P_P_R","CA_P_P_R","NM_P_P_R","TX_P_P_R"))

fig=plt.figure("renewable energy/total energy(R_T)")
fig.set(alpha=0.2)
x1 =AZ_Data["Year"]
y1 = AZ_Data["R_T"]
x2= x1
y2=CA_Data["R_T"]
x3=x1
y3=NM_Data["R_T"]
x4=x1
y4=TX_Data["R_T"]
plt.plot(x1,y1,x2,y2,x3,y3,x4,y4)
plt.plot(kind='kde')
plt.xlabel("year")
plt.ylabel("renewable energy/total energy(R_T)")
plt.title("line chart showing the share of renewable energy in total energy consumption over the year")
plt.xticks(np.arange(1960, 2010, 1), fontsize=5)
plt.legend(("AZ_R_T","CA_R_T","NM_R_T","TX_R_T"))

fig=plt.figure("R_N_T")
fig.set(alpha=0.2)
x1 =AZ_Data["Year"]
y1 = AZ_Data["R_N_T"]
x2= CA_Data["Year"]
y2=CA_Data["R_N_T"]
x3=NM_Data["Year"]
y3=NM_Data["R_N_T"]
x4=TX_Data["Year"]
y4=TX_Data["R_N_T"]
plt.plot(x1,y1,x2,y2,x3,y3,x4,y4)
plt.plot(kind='kde')
plt.xlabel("year")
plt.ylabel("R_N_T")
plt.title("line chart for renewable energy production of each state over the year")
plt.xticks(np.arange(1960, 2010, 1), fontsize=5)
plt.legend(("AZ_R_N_T","CA_R_N_T","NM_R_N_T","TX_R_N_T"))
plt.show()

df=pd.DataFrame(columns=["a","b"])
print(df)
df.loc[df.shape[0]+1]={"a":1,"b":2}
print(df)



