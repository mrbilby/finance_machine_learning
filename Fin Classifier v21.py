import bs4 as bs
import math
from collections import Counter
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

style.use('ggplot')
global runningTotal
runningTotal = 0
global runningCount
runningCount = 0

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/FTSE_100_Index')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text
        tickers.append(ticker)
    print(tickers)    
    with open("FTSE100tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return tickers


def get_data_from_google():
    df=pd.read_csv('lse list.csv')
    tickers=df['TIDM'].tolist()
    tickerlist = []

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2010, 1, 1)
    end = dt.datetime(2017, 9, 7)
    
    for ticker in tickers:
        try:
            # just in case your connection breaks, we'd like to save our progress!
            if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
                df = web.DataReader(ticker, "google", start, end)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
            else:
                print('Already have {}'.format(ticker))
            print(ticker + " complete")
            tickerlist.append(ticker)

        except:
            print(str(ticker) + " failed")
            pass
    print(tickerlist)
    ticks = pd.DataFrame(tickerlist, columns=['Tickers'])
    ticks.to_csv('lsetickers.csv')
    
def compile_data():
    with open("FTSE100tickers.pickle","rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()
    
    for count,ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Close':ticker}, inplace=True)
        df.drop(['Open','High','Low','Volume'],1,inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)
    print(main_df.head())
    main_df.to_csv('ftse100_joined_google_closes.csv')

def stock_remapper():
    df=pd.read_csv('ftse100_joined_google_closes.csv')
    df = df.drop('Date',1)
    df.fillna(0, inplace=True)
    for column in df:
        print(column)
        i = 0
        for index, row in df.iterrows():
            percUplift = 1.01
            percDowner = 0.98
##            print(row[column])
            saveIter = row[column]

            if i>1:
                if row[column]==0:
                    row[column]=0
                elif lastIter == 0:
                    row[column]=0
                elif (row[column]/lastIter)>percUplift:
                    row[column]=1
                elif (row[column]/lastIter)<percDowner:
                    row[column]=-1
                else:
                    row[column]=0
            lastIter=saveIter
##            print(row[column])
            i=i+1
    df = df[2:]
    df.to_csv("remapped stocks2.csv", index = False)

def stock_classifier(ticker):
    lag=1
    cutoff = 1700
    df=pd.read_csv('remapped stocks2.csv')
    origPrice = df[ticker]
    df[ticker]=df[ticker].shift(-lag)
    df = df[:-lag]
    y = np.array(df[ticker])
    X = np.array(df.drop([ticker], 1))
    X_train = X[:cutoff]
    X_test = X[cutoff:]
    y_train = y[:cutoff]
    y_test = y[cutoff:]
##    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    
##    clf = VotingClassifier([('knn',neighbors.KNeighborsClassifier()),('rfor',RandomForestClassifier())])
    clf = RandomForestClassifier(n_estimators = 50)
##    clf = MLPClassifier(alpha=1)
##    clf=neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(confidence)
    predPrice = clf.predict(X_test)
    
    decBuySell=[]
    y=0
    output = 0
    for x in predPrice:
        if x == 1:
            decBuySell.append("buy")
##            print(ticker+" Buy")
            output = 1
        elif x == -1:
            decBuySell.append("sell")
##            print(ticker+" Sell")
            output = 1
        else:
            decBuySell.append("hold")
        y=y+1
    origdf=pd.read_csv('ftse100_joined_google_closes.csv')
    newdf = origdf[['Date', ticker]].copy()
    newdf = newdf[cutoff+(lag+2):]
##    print(len(newdf.index))
##    print(len(decBuySell))
    newdf['Prediction']=decBuySell
    closes = newdf[ticker].tolist()
    buyrate=0
    counter = 0
    rollingStock = []
    rollingCash = []
    current = 0
    stock = 0
    cash = 1000
    for pos in decBuySell:
        if pos == "buy":
            if np.isfinite(closes[counter]):
                if current == 0:
                    buyrate=buyrate+1
                    current = 1
                    stock = float(cash)/(closes[counter])
                    cash = 0
        elif pos == "sell":
            if np.isfinite(closes[counter]):
                if current==1:
                    current=0
                    cash = stock*closes[counter]
                    stock = 0
        rollingStock.append(stock)
        rollingCash.append(cash)
        counter = counter+1
    newdf['rollingStock']=rollingStock
    newdf['rollingCash']=rollingCash
    if cash == 0:
        if np.isfinite(closes[-1]):   
            finalPos = (stock*closes[-1])
            print("Final Position: " + str(finalPos))
        else:
            for k in reversed(closes):
                if math.isnan(k) == False:
                    finalPos = (stock*k)
                    print("Final Position: " + str(finalPos))
                    break
    else:
        print("Final Position: " +str(cash))
        finalPos = cash
    if cash != 1000:
        newdf.to_csv('stock_dfs/{}.csv'.format("Classified" + ticker))
        global runningTotal
        runningTotal = runningTotal+finalPos-1000
        global runningCount
        runningCount = runningCount+1
        print(runningTotal)        
    if buyrate>3:
        print("over three buys " + ticker)


def multi_check():
    with open("FTSE100tickers.pickle","rb") as f:
        tickers = pickle.load(f)
    for count,ticker in enumerate(tickers):
        print("\n"+ticker)
        stock_classifier(ticker)
    global runningTotal
    global runningCount
    avgPos = runningTotal/runningCount
    print("Absolute position: " + str(runningTotal))
    print("Average position: " +str(avgPos))
##
##    origPrice = df[ticker]
##    df[ticker]=df[ticker].shift(-lag)
##    df = df[:-lag]
##    y = np.array(df[ticker])
##    X = np.array(df.drop([ticker], 1))
##    print(df.head())
##    print(df.tail())

##stock_remapper()
##multi_check()

get_data_from_google()

