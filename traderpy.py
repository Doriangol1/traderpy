
import yfinance as yf
import pandas as pd

ticker = yf.Ticker('nxst')
incomeSt = ticker.income_stmt
balanceS = ticker.balance_sheet
cashF = ticker.cashflow
tickInfo = ticker.info
curPrice = tickInfo['currentPrice']
EV = tickInfo['enterpriseValue']
print ("EV ", EV)
netDebt = balanceS.iloc[3, 0]
print ("debt ", netDebt)
Shares = tickInfo['sharesOutstanding']
print ("Shares ", Shares)

def StockVal(curPrice, EV, netDebt, Shares):
    print("Current Price: $", curPrice)
    EquityV = EV - netDebt  
    EqVperShare = EquityV / Shares
    print("Estimated Equity Value Per Share:  $" , EqVperShare)
    upside = ((EqVperShare-curPrice) / curPrice) * 100
    
    if (upside > 0):
        print("Upside: %",  upside)
        print("stock may be undervalued at the moment")
    else :
        print("Downside: %",  upside)
        print("stock may be overvalued at the moment")

StockVal(curPrice, EV, netDebt, Shares)
#https://chandlerzuo.github.io/blog/2017/11/darnn
