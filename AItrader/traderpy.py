
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
import tensorflow
from tensorflow import keras
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
plt.style.use('seaborn-darkgrid')
warnings.filterwarnings('ignore')


class MyTicker:
    def __init__(self, name):
        self.ticker = yf.Ticker(name)
        self.incomeSt = self.ticker.income_stmt
        self.balanceS = self.ticker.balance_sheet
        self.cashF = self.ticker.cashflow
        self.tickInfo = self.ticker.info
        self.curPrice = self.tickInfo['currentPrice']
        self.EV = self.tickInfo['enterpriseValue']
        self.netDebt = self.balanceS.iloc[3, 0]
        self.Shares = self.tickInfo['sharesOutstanding']
        self.MktCap = self.tickInfo['marketCap']
        self.trailingPE = self.tickInfo['trailingPE']
        self.Ebitda = self.tickInfo['ebitda']
        self.DE = self.tickInfo['debtToEquity']
        self.currentR = self.tickInfo['currentRatio']
        self.quickR = self.tickInfo['quickRatio']
        self.fcf = self.tickInfo['freeCashflow']
        self.totalD = self.tickInfo['totalDebt']

#symbol = input("TYPE SYMBOL: ")
#userTicker = MyTicker(symbol)

def StockVal(curPrice, EV, netDebt, Shares):
    print("Current Price: $", curPrice)
    EquityV = EV - netDebt  
    EqVperShare = EquityV / Shares
    print("Estimated Equity Value Per Share:  $" , EqVperShare)
    upside = ((EqVperShare-curPrice) / curPrice) * 100
    
    if (upside > 0):
        print("Upside: %",  upside)
        print("stock may be undervalued at the moment")
    else:
        print("Downside: %",  upside)
        print("stock may be overvalued at the moment")





def industryAnalysis (userTicker):
    
    numTickers = int(input ('how many companies would u like to compare? (integer) '))
    Tickers = []
    Tickers.append(userTicker)
    for i in range(numTickers):
        sym = input ("symbol: ")
        Tickers.append(MyTicker(sym))
    data = []
    indexList = []
    for k in range(len(Tickers)):
        curTicker = Tickers[k]
        curRow = {'curPrice': [curTicker.curPrice], 'MktCap': [curTicker.MktCap], 'trailingPE': [curTicker.trailingPE], 'EV': [curTicker.EV], 'Ebitda': [curTicker.Ebitda], 'netDebt': [curTicker.netDebt], 'DE': [curTicker.DE], 'quickR': [curTicker.quickR], 'currentR': [curTicker.currentR], 'fcf': [curTicker.fcf], 'totalD': [curTicker.totalD]}
        data.append(curRow)
        indexList.append(curTicker.tickInfo['symbol'])
    df = pd.DataFrame((data), indexList)
    print(df)

# StockVal(userTicker.curPrice, userTicker.EV, userTicker.netDebt, userTicker.Shares)
# industryAnalysis(userTicker)

#credit to geekforgeeks


def svcClass(df, X_train, Y_train, X_test, X, Y, Y_test): 
     
    #Support vector classifier (SVC)
    cls = SVC().fit(X_train, Y_train)

    #sell or buy predictions
    df['Predicted_Signal'] = cls.predict(X)

    #calculate daily returns
    df['Return'] = df.Close.pct_change() * 100

    #calculate strategy returns
    df['Strategy_Return'] = df.Return *df.Predicted_Signal.shift(1)

    #calculate cumulative returns
    df['Cum_Ret'] = df['Return'].cumsum()

    #plot strat cumulative returns
    df['Cum_Strategy'] = df['Strategy_Return'].cumsum()

    
    
    print(df)

    print(df.isnull().sum())
    
    #fill na values with means (last day will always be na)
    df.fillna(value = df.mean(), inplace = True)
    print(df)

    plt.plot(df['Cum_Ret'], color = 'red')
    plt.plot(df['Cum_Strategy'], color = 'blue')
    plt.show()
   
    #modification for accuracy training (dont think this works)
    cumRetNumpy = df['Cum_Ret'].to_numpy()
    cumStrategyNumpy = df['Cum_Strategy'].to_numpy()
    rtrnNumpy = df['Return'].to_numpy()
    strategyRtrnNumpy = df['Strategy_Return'].to_numpy
    print(df['Cum_Ret'])
    print(cumRetNumpy)
    print(df['Cum_Strategy'])
    print(cumStrategyNumpy)

    #accuracy
    print('Accuracy:', cls.score(X_test, Y_test))

def LSTMPred (df, X_train, X_test, Y_train, Y_test, X, Y):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units = 1, return_sequences=True, input_shape = (X_train.shape[1], 1)))
    model.add(keras.layers.LSTM(units = 1))
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dropout(.5))
    model.add(keras.layers.Dense(1))
    print(model.summary)

    model.compile(optimizer='adam', loss = 'mean_squared_logarithmic_error')
    StandardScaler().fit_transform(X_train)
    model.fit(X_train, Y_train, epochs = 10)
    pred = model.predict(X)
    df['LSTM'] = pred
    plt.plot(df['LSTM'], color = 'yellow')
    plt.show()
    #accuracy
    print('Accuracy:', model.evaluate(X_test, Y_test))
def stockPredAI (filename):

    #load data
    pd.options.display.float_format = '{:,.4f}'.format 
    df = pd.read_csv(filename)
    print(df)
    
    #feature engineering
    #add quarter end(from different geekforgeeks project) 
    splitted = df['Date'].str.split('-', expand=True)
    df['Year'] = splitted[0].astype('int')
    df['Month'] = splitted[1].astype('int')
    df['Day'] = splitted[2].astype('int')
    df['quarter_end'] = np.where(df['Month']%3 == 0, 1, 0)
    print(df)

    #change index columns to index
    df.index = pd.to_datetime(df['Date'])

    #drop original date column
    df = df.drop(['Date'], axis = 'columns')
    print(df)

    #predictor variables
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
    print(df.head())

    #store predictor variables in X
    X = df[['Open-Close','High-Low', 'quarter_end', 'Volume']]
    print(X.head())

    #define target variables
    Y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    print(Y)

    split_percentage = .8
    split = int(split_percentage*len(df))

    #train
    X_train = X[:split]
    Y_train = Y[:split]

    #test
    X_test = X[split:]
    Y_test = Y[split:]
   
    svcClass(df, X_train, Y_train, X_test, X, Y, Y_test)
    LSTMPred(df, X_train, X_test, Y_train, Y_test, X, Y)

   
stockPredAI('AAPL.csv')


