def optimi_portfo(x=10000):
    df = pd.read_csv('stock_price.csv',index_col='Date',parse_dates=True)
    
    potfolio_weight_list = []
    potfolio_return_list = []
    potfolio_volitility_list = []
    potfolio_sharpe_list = []
    
    for times in range(x):
        #隨機權重
        random_weight = np.array(np.random.random(len(df.columns)))
        random_weight = random_weight / np.sum(random_weight)
        potfolio_weight_list.append(random_weight)
        #計算投組報酬率
        log_ret = np.log(df/df.shift(1))
        r = np.sum((log_ret.mean() * random_weight) *252)
        potfolio_return_list.append(r)
        #計算投組報酬率的變異數
        exp_vol = np.sqrt(np.dot(random_weight.T, np.dot(log_ret.cov() * 252, random_weight)))
        potfolio_volitility_list.append(exp_vol)
        #計算夏普值
        potfolio_sharpe_list.append(r/exp_vol)
        
    max_ = np.array(potfolio_sharpe_list).argmax()
    max_sharp = potfolio_sharpe_list[max_]
    max_sharp_return,max_sharp_return_vol = potfolio_return_list[max_],potfolio_volitility_list[max_]
    best_weight = potfolio_weight_list[max_]
    
    potfo_wight={}
    for i in range(len(df.columns)):
        if df.columns[i] not in potfo_wight:
            potfo_wight[df.columns[i]] = best_weight[i]*100
    sort_weight = sorted(potfo_wight.items(),key=lambda x:x[1],reverse=True)
    for i in sort_weight:
        print('股票代號{},投資權重為{:.2f}%'.format(i[0],i[1]))
        
    plt.figure(figsize=(12,8))
    plt.scatter(potfolio_volitility_list,potfolio_return_list,c=potfolio_sharpe_list,cmap='plasma')
    plt.scatter(max_sharp_return_vol,max_sharp_return,c='red',s=100,edgecolors='black')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.show()

##############################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

optimi_portfo(50000)