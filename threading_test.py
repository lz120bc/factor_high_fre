import threading
import pandas as pd
import talib as ta
import time
from queue import Queue

data = pd.read_csv('E:\\data\\tick.csv', low_memory=False)
num = []
lock = threading.Lock()


def count_price(g: pd.DataFrame):
    da = g[['date', 'time']].copy()
    da['price'] = (g['bid_price1'] + g['offer_price1']) / 2
    da['price_ma'] = ta.STDDEV(da['price'], 20)
    global num
    lock.acquire()
    num.append(da)
    lock.release()
    # q.put(da)


if __name__ == "__main__":
    threads = []
    start_time = time.process_time()
    que = Queue()
    for _, groups in data.groupby('securityid'):
        thread = threading.Thread(target=count_price, args=(groups,))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    # for i in range(len(que.queue)):
    #     num.append(que.get())
    end_time = time.process_time()
    print(end_time - start_time)
