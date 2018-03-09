import pandas as pd
def chartRNT(data,state):
    i = data.loc[(data['MSN'] == 'TETCB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TETCB_Data"})
    j = data.loc[(data['MSN'] == 'NUETB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "NUETB_Data"})
    z = pd.merge(i, j, how='outer')
    j = data.loc[(data['MSN'] == 'RETCB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "RETCB_Data"})
    z = pd.merge(z, j, how='outer')
    R_N_T=[]
    for k in range(len(z.index)):
        R_N_T.append(1.0*(z.iat[k,2]+z.iat[k,3])/z.iat[k,1])
    z['R_N_T']=R_N_T
    j = data.loc[(data['MSN'] == 'TPOPP') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TPOPP_Data"})
    z = pd.merge(z, j, how='outer')
    j = data.loc[(data['MSN'] == 'GDPRX') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "GDPRX_Data"})
    z = pd.merge(z, j, how='outer')
    j = data.loc[(data['MSN'] == 'NUETD') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "NUETD_Data"})
    z=pd.merge(z,j,how='outer')
    indus = data.loc[(data['MSN'] == 'TEICB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TEICB/TETCB"})
    cemr = data.loc[(data['MSN'] == 'TECCB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TECCB/TETCB"})
    rent = data.loc[(data['MSN'] == 'TERCB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TERCB/TETCB"})
    atr = data.loc[(data['MSN'] == 'TEACB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TEACB/TETCB"})
    i_TETCB = []
    c_TETCB = []
    r_TETCB = []
    a_TETCB = []
    for i in range(len(z.index)):
        i_TETCB.append(1.0 * indus.iat[i, 1] / z.iat[i, 1])
        c_TETCB.append(1.0 * cemr.iat[i, 1] / z.iat[i, 1])
        r_TETCB.append(1.0 * rent.iat[i, 1] / z.iat[i, 1])
        a_TETCB.append(1.0 * atr.iat[i, 1] / z.iat[i, 1])
    z["i_TETCB"] = i_TETCB
    z["c_TETCB"] = c_TETCB
    z["r_TETCB"] = r_TETCB
    z["a_TETCB"] = a_TETCB
    j = data.loc[(data['MSN'] == 'PATCD') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "PATCD_Data"})
    z = pd.merge(z, j, how='outer')
    j = data.loc[(data['MSN'] == 'NGTCD') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "NGTCD_Data"})
    z = pd.merge(z, j, how='outer')
    j = data.loc[(data['MSN'] == 'CLTCD') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "CLTCD_Data"})
    z = pd.merge(z, j, how='outer')
    return z
def find_TETCB(data,state):
    i = data.loc[(data['MSN'] == 'TETCB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TETCB_Data"})
    j = data.loc[(data['MSN'] == 'TETCD') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TETCD_Data"})
    z = pd.merge(i, j, how='outer')
    j = data.loc[(data['MSN'] == 'TPOPP') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TPOPP_Data"})
    z = pd.merge(z, j, how='outer')
    j = data.loc[(data['MSN'] == 'GDPRX') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "GDPRX_Data"})
    z = pd.merge(z, j, how='outer')
    indus = data.loc[(data['MSN'] == 'TEICB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TEICB/TETCB"})
    cemr = data.loc[(data['MSN'] == 'TECCB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TECCB/TETCB"})
    rent = data.loc[(data['MSN'] == 'TERCB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TERCB/TETCB"})
    atr = data.loc[(data['MSN'] == 'TEACB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TEACB/TETCB"})
    i_TETCB = []
    c_TETCB = []
    r_TETCB = []
    a_TETCB = []
    for i in range(len(z.index)):
        i_TETCB.append(1.0 * indus.iat[i, 1] / z.iat[i, 1])
        c_TETCB.append(1.0 * cemr.iat[i, 1] / z.iat[i, 1])
        r_TETCB.append(1.0 * rent.iat[i, 1] / z.iat[i, 1])
        a_TETCB.append(1.0 * atr.iat[i, 1] / z.iat[i, 1])
    z["i_TETCB"] = i_TETCB
    z["c_TETCB"] = c_TETCB
    z["r_TETCB"] = r_TETCB
    z["a_TETCB"] = a_TETCB
    j = data.loc[(data['MSN'] == 'PATCD') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "PATCD_Data"})
    z = pd.merge(z, j, how='outer')
    j = data.loc[(data['MSN'] == 'NGTCD') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "NGTCD_Data"})
    z = pd.merge(z, j, how='outer')
    j = data.loc[(data['MSN'] == 'CLTCD') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "CLTCD_Data"})
    z = pd.merge(z, j, how='outer')


    return z


def find_RT(data,state):
    i = data.loc[(data['MSN'] == 'TETCB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TETCB_Data"})
    j = data.loc[(data['MSN'] == 'RETCB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "RETCB_Data"})
    z = pd.merge(i, j, how='outer')
    R_T = []
    for k in range(len(z.index)):
        R_T.append(1.0 * (z.iat[k, 2]) / z.iat[k, 1])
    z['R_T'] = R_T
    j = data.loc[(data['MSN'] == 'TPOPP') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TPOPP_Data"})
    z = pd.merge(z, j, how='outer')
    j = data.loc[(data['MSN'] == 'GDPRX') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "GDPRX_Data"})
    z = pd.merge(z, j, how='outer')
    # j = data.loc[(data['MSN'] == 'NUETD') & (data['StateCode'] == state), ["Year", "Data"]].rename(
    #     columns={"Data": "NUETD_Data"})
    # z = pd.merge(z, j, how='outer')
    indus = data.loc[(data['MSN'] == 'TEICB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TEICB/TETCB"})
    cemr = data.loc[(data['MSN'] == 'TECCB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TECCB/TETCB"})
    rent = data.loc[(data['MSN'] == 'TERCB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TERCB/TETCB"})
    atr = data.loc[(data['MSN'] == 'TEACB') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "TEACB/TETCB"})
    i_TETCB = []
    c_TETCB = []
    r_TETCB = []
    a_TETCB = []
    for i in range(len(z.index)):
        i_TETCB.append(1.0 * indus.iat[i, 1] / z.iat[i, 1])
        c_TETCB.append(1.0 * cemr.iat[i, 1] / z.iat[i, 1])
        r_TETCB.append(1.0 * rent.iat[i, 1] / z.iat[i, 1])
        a_TETCB.append(1.0 * atr.iat[i, 1] / z.iat[i, 1])
    z["i_TETCB"] = i_TETCB
    z["c_TETCB"] = c_TETCB
    z["r_TETCB"] = r_TETCB
    z["a_TETCB"] = a_TETCB
    j = data.loc[(data['MSN'] == 'PATCD') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "PATCD_Data"})
    z = pd.merge(z, j, how='outer')
    j = data.loc[(data['MSN'] == 'NGTCD') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "NGTCD_Data"})
    z = pd.merge(z, j, how='outer')
    j = data.loc[(data['MSN'] == 'CLTCD') & (data['StateCode'] == state), ["Year", "Data"]].rename(
        columns={"Data": "CLTCD_Data"})
    z = pd.merge(z, j, how='outer')
    return z
state=["AZ","CA","NM","TX"]
data=pd.read_excel("D:\meisai\cdata.xlsx")



a=find_TETCB(data,"AZ")
a.to_csv("D:\meisai\AZTETCB.csv")
b=find_TETCB(data,"CA")
b.to_csv("D:\meisai\CATETCB.csv")
c=find_TETCB(data,"NM")
c.to_csv("D:\meisai\jNMTETCB.csv")
d=find_TETCB(data,"TX")
d.to_csv("D:\meisai\TXTETCB.csv")

a=chartRNT(data,"AZ")
a.to_csv("D:\meisai\AZRNT.csv")
b=chartRNT(data,"CA")
b.to_csv("D:\meisai\CARNT.csv")
c=chartRNT(data,"NM")
c.to_csv("D:\meisai\jNMRNT.csv")
d=chartRNT(data,"TX")
d.to_csv("D:\meisai\TXRNT.csv")

a=find_RT(data,"AZ")
a.to_csv("D:\meisai\AZRT.csv")
b=find_RT(data,"CA")
b.to_csv("D:\meisai\CART.csv")
c=find_RT(data,"NM")
c.to_csv("D:\meisai\jNMRT.csv")
d=find_RT(data,"TX")
d.to_csv("D:\meisai\TXRT.csv")
