# from sympy import *
# k=symbols('k')
# i=solve(((0.35*k*4811760.808+12741658.06*0.67*k+1352551.103*0.21*k+0.59*1352551.103*k)/(4811760.808+12741658.06+1352551.103
#                                                                                      +1352551.103)-0.55),k)
# print('2050年',0.35*i[0],'   ',0.67*i[0],'   ',0.21*i[0],'   ',0.59*i[0])
#
# k=solve(((0.35*k*2467999.345+0.67*k*9999887.899+0.21*k*929167.7529+0.59*k*15579789.39)/(2467999.345+9999887.899+929167.7529
#                                                                                         +15579789.39)-0.4),k)
# print('2025年',0.35*k[0],'   ',0.67*k[0],'   ',0.21*k[0],'   ',0.59
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def paint(data,state):
    fig=plt.figure(state+'Energy  diagram between 1960 and 2009')
    plt.title(state+'Energy production diagram between 1960 and 2009')
    plt.plot(kind='kde')
    plt.xlabel("year")
    plt.ylabel('Energy values')
    plt.xticks(np.arange(1960, 2010, 1), fontsize=5)
    plt.plot(data['Year'],data['TETCB_Data'],data['Year'],data['RETCB_Data'],data['Year'],data['P_P_R'],data['Year'],data['NUETB_Data'])
    plt.legend(("TETCB_Data", "RETCB_Data", "P_P_R", "NUETB_Data"))
    fig.set_size_inches(10,8)
    fig.savefig(state+'P_P_R等能源.png')

    # fig=plt.figure(state+'Energy  proportion diagram between 1960 and 2009')
    # plt.title(state+'Energy  proportion diagram between 1960 and 2009')
    # plt.plot(kind='kde')
    # plt.xlabel("year")
    # plt.ylabel('Energy proportion values')
    # plt.xticks(np.arange(1960, 2010, 1), fontsize=5)
    # plt.plot(data['Year'], 1.0*data['PAPRB_Data']/(data['PAPRB_Data']+data['NGMPB_Data']+data['CLPRB_Data']), data['Year'], 1.0*data['NGMPB_Data']/(data['PAPRB_Data']+data['NGMPB_Data']+data['CLPRB_Data']),
    #          data['Year'], 1.0*data['CLPRB_Data']/(data['PAPRB_Data']+data['NGMPB_Data']+data['CLPRB_Data']))
    # plt.legend(("PAPRB_Data", "NGMPB_Data", "CLPRB_Data"))
    # fig.set_size_inches(10, 8)
    # plt.savefig(state+'三种生产能源占比图.png')

datapath='D:\meisai'
state=['AZ','CA','NM','TX']
for i in state:
    if i!='NM':
        data = pd.read_csv(datapath +"\\"+ i+'_Data_ALL.csv')
        paint(data, i)
    else:
        data = pd.read_csv(datapath + "\\" + 'h'+i + '_Data_ALL.csv')
        paint(data, i)










