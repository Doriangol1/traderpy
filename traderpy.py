
import yfinance as yf
import pandas as pd

import torch
from torch import nn
from torch.autograd import Variable

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

symbol = input("TYPE SYMBOL: ")
userTicker = MyTicker(symbol)

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

StockVal(userTicker.curPrice, userTicker.EV, userTicker.netDebt, userTicker.Shares)
industryAnalysis(userTicker)
#https://chandlerzuo.github.io/blog/2017/11/darnn
