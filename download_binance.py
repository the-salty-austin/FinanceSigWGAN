import requests
import time
import pandas as pd
import datetime

url = "https://www.binance.com/api/v3/klines"

def get_data_since(symbol, startTime, endTime, interval):
    '''
    symbol: BTC, ETH, BNB... \n
    startTime: when to start \n
    endTime: last row of output is no earlier than it\n
    interval: '1m', '4h', '1d'
    '''
    if interval=="1m":
        s = 60*1*1
    if interval=="30m":
        s = 60*30*1
    elif interval=="1h":
        s = 60*60*1
    elif interval=="4h":
        s = 60*60*4
    elif interval=="1d":
        s = 60*60*24

    data = []

    unixTimeNow = startTime.timestamp()*1000

    while unixTimeNow <= endTime.timestamp()*1000:
        params = {
            'symbol': f'{symbol}USDT',
            'interval': interval,
            'limit': '1000',
            'startTime': int( unixTimeNow ),
        }

        price = requests.get(url, params=params).json()

        if "code" in price:
            if price["code"] == "-1121":
                print(f'{symbol} Invalid symbol.')
            print(symbol, price)
            # unixTimeNow += 1000 * s * 1000
            # time.sleep(0.8)
            break
        # print( price, type(price) )
        '''
        Response Format
            [
                [
                    1499040000000,      // Kline open time
                    "0.01634790",       // Open price
                    "0.80000000",       // High price
                    "0.01575800",       // Low price
                    "0.01577100",       // Close price
                    "148976.11427815",  // Volume
                    1499644799999,      // Kline Close time
                    "2434.19055334",    // Quote asset volume
                    308,                // Number of trades
                    "1756.87402397",    // Taker buy base asset volume
                    "28.46694368",      // Taker buy quote asset volume
                    "0"                 // Unused field, ignore.
                ]
            ]
        '''
        for row in price:
            # time, open, high, low, close, volume
            t, o, h, l, c, v = int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
            unixTimeNow = t
            data.append( [t, o, h, l, c, v] )
            # print(t)
            # print( [t, o, h, l, c, v] )

        unixTimeNow += s*1000

        if len(price) == 0:
            time.sleep(1.3)
        else:
            time.sleep(0.8)

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.timestamp = pd.to_datetime(df.timestamp, unit='ms')

    return df

if __name__ == "__main__":
    symbols = ["BTC", "ETH", "BNB"]
    period = "1h"
    start = datetime.datetime(2021, 1,  1, 0, 0)
    end   = datetime.datetime(2023, 7, 27, 0, 0)

    for i, symbol in enumerate( symbols[:] ):
        
        print(f"Begin downloading: H1-{symbol}")
        df = get_data_since(symbol, start, end, period )
        print(f"Download completed: H1-{symbol}")
        print(f"There are {df.close.isna().sum() } N/As in close prices.")

        df.to_csv(f"./datasets/MyBinance/{symbol}USDT_{period}.csv")