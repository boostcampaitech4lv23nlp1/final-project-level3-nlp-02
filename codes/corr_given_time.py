import pandas as pd
import time
from datetime import datetime, timedelta
import urllib.request
import yfinance as yf
import json
from tqdm import tqdm
from collections import deque

# KOSPI 가격 불러오는 데이터
def StockPrice(start_date, end_date):
    '''
    start_date : "YYYY-MM-DD" (str)
    end_date : "YYYY-MM-DD" (str)
    '''
    # 종목, 날짜 설정
    stock_name = "^KS11"
    start = start_date
    end = end_date
    ticker = yf.Ticker(stock_name)

    # yfinance의 경우 start~end-1의 데이터를 가져와서 price_end로 1일을 늘려준다
    price_end = str(pd.Timestamp(end).date() + timedelta(1))

    # start~end-1 까지의 데이터를 가져온다.
    df = ticker.history(interval='1d', start=start,
                        end=price_end, auto_adjust=False)
    # 인덱스를 새로 만들고, 기존 인덱스였던 datetimeindex를 Date라는 column으로 만들기, #dataframe, series에서 corr 구할때 동일한 index끼리 대응 시켜 비교하기 떄문에 해당 과정이 필요!
    df.reset_index(inplace=True)
    # Date가 timestamp 형식이었는데 일반적인 date 형식으로 바꾸기
    df["Date"] = df["Date"].apply(lambda x: x.date())

    # Date, Close 만 남기고 제거
    price_df = df.drop(["Open", "High", "Low", "Adj Close",
                       "Volume", "Dividends", "Stock Splits"], axis=1)

    return price_df


def get_search_volume(timeUnit="date"):
    '''
    start : "YYYY-MM-DD"
            2016-01-01 부터 가능
    end : "YYYY-MM-DD"
    timeUnit : "date" or "week" or "month"
    
    # 가능한 환경
    device : pc/mobile/all 설정 가능
    ages : 연령대 설정 가능
    gender : 
    
    dictionary 리턴 -> 필요하면 json으로.
    '''
    # naver_api 세팅
    client_id = "HWwt3Hc5Nzgt4WVgr3Rd"
    client_secret = "pMqb1CX0nT"
    url = "https://openapi.naver.com/v1/datalab/search"
    
    # 검색 내용 설정
    start = '2016-01-01'
    # end = datetime.today().strftime("%Y-%m-%d")
    end = "2023-01-20"
    datelist = [date.strftime("%Y-%m-%d") for date in pd.date_range(start, end)]
    dict_result = {}
    keywords = pd.read_csv(
        "./data/final_candi_list.csv", index_col=0)['keyword']
    for key in keywords:
        body = {
                "startDate" : start,
                "endDate" : end,
                "timeUnit" : timeUnit,
                "keywordGroups":[
                    {"groupName":key, "keywords": [key]}
                ],
            }
        body = json.dumps(body)

        # 검색량 불러오기 -> json 파일을 읽어(a) dict로(b) 리턴
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id",client_id)
        request.add_header("X-Naver-Client-Secret",client_secret)
        request.add_header("Content-Type","application/json")
        response = urllib.request.urlopen(request, data=body.encode("utf-8"))
        rescode = response.getcode()
        if(rescode==200):
            response_body = response.read()
            a = response_body.decode('utf-8')
            b= json.loads(a)
        else:
            print("Error Code:" + rescode)

        # DataFrame 만들기
        # 검색량 0인 부분 0.0으로 채워넣기
        D = deque([d["period"] for d in b['results'][0]['data']])
        SV = deque([d["ratio"] for d in b['results'][0]['data']])
        setD = set(D)

        l = []
        for date in datelist:
            if date in setD: 
                d = D.popleft(); sv = SV.popleft()
                l.append({'period':d, 'ratio':sv})
            else:
                l.append({'period':date, 'ratio':0.0})

        date = [p['period'] for p in l]
        ratio = [r['ratio'] for r in l]

        dict_tmp = {key : {'Date' : date, 'Search Volume' : ratio}}
        dict_result.update(dict_tmp)
    
    # 함수를 수행할 dags 디렉토리를 기준으로 상대경로 시작.
    with open('./data/final_candi_search_volume.json', 'w', encoding="UTF-8") as fp:
        json.dump(dict_result, fp, ensure_ascii=False, indent = 4)
    
    return dict_result


def searchVolume2(key_name, dictdata, start, end, timeUnit="date"):
    '''
    key_name : keywords list -> ['KOSPI', 'KOSDAQ']
    start : "YYYY-MM-DD" 
            2016-01-01 부터 가능
    end : "YYYY-MM-DD"
    timeUnit : "date" or "week" or "month"

    # 가능한 환경
    device : pc/mobile/all 설정 가능
    ages : 연령대 설정 가능
    gender : 
    '''
    from collections import deque
    import os

    b = dictdata

    # DataFrame 만들기
    d = b[f"{key_name[0]}"]['Date']
    sv = b[f"{key_name[0]}"]['Search Volume']
    s_i, e_i = d.index(start), d.index(end)

    df = pd.DataFrame({'Date': d[s_i:e_i+1], 'Search Volume': sv[s_i:e_i+1]})

    return df


def TLCC(price_df, key_name, start, end, lag, data):
    '''
    price_df: 코스피가격 불러와서 적당히 처리하고 저장한 dataframe(Date가 str임에 주의)
    key_name: [candi_key 이름](list)
    start: 시작날짜(str) ex> "2020-01-01"
    end: 끝날짜(str) ex> "2020-12-31"
    lag: correlation 구할때의 시차(int)
    data: volume_dict(dict)
    '''

    # 1. price_df -> start~end까지의 stock_name의 가격데이터 가져오기
    # column = (Date, Close)

    # 2. search_volume_df -> start~end까지의 key_name의 검색량 데이터 가져오기
    # column = (Date, Search Volume)
    # 승엽이가 준 함수 사용

    # 넉넉히 start 대비 lag만큼 앞서서 가져온다
    search_volume_start = str(pd.Timestamp(start).date() - timedelta(lag))
    search_volume_df = searchVolume2(key_name, data, search_volume_start, end)

    # 3. price_df의 날짜에서 lag만큼 뺸 날짜에 해당하는 데이터만 search_volume_df에서 가져와서 search_volume_lagged_df 만들기
    lag = lag

    # price_df의 Date에서 lag만큼 뺀뒤, 해당 값들을 str으로 만들어서 time이라는 리스트에 집어 넣자
    time = list(map(str, (price_df["Date"]-timedelta(lag)).values))

    # search_volume_df에서 Date가 time에 해당하는 행만 뽑아서 search_volume_lagged_df 만들기!
    search_volume_lagged_df = search_volume_df[search_volume_df["Date"].isin(
        time)].reset_index(drop=True)
    # print(search_volume_lagged_df)
    # print("---------"*5)
    # print("---------"*5)
    # 4. correlation 구하기
    corr = price_df["Close"].corr(search_volume_lagged_df["Search Volume"])

    return corr


def corr_given_time(start_date, end_date):
    '''
    keywords : 검색량 불러올 키워드 리스트 (List)
    start : 주가 시작 날짜 ex. "2016-02-01" (str)
    end : 주가 끝 날짜 ex. "2023-01-26" (str)
    '''
    # final_candi_list 불러오기.
    # 모델이 바뀌면 csv파일을 바꿔줘야 함
    keywords = pd.read_csv(
        "./data/final_candi_list.csv", index_col=0)['keyword']

    # 검색량 불러오기 (dict)
    with open(f"./data/final_candi_search_volume.json", "r") as json_file:
        search_volume = json.load(json_file)

    # 주가 불러오기
    # 코스피에 대한 DataFrame -> columns : ['Date', 'Close']
    price_df = StockPrice(start_date, end_date)

    # correlation 계산하기
    # corr 계산
    result = []
    for key in tqdm(keywords, leave=True):
        corr_by_lag = []
        max_abs_corr = -100  # 임의의 작은 값으로 세팅
        max_lag = -100  # 임의의 값으로 세팅
        for lag in range(1, 11):
            corr = TLCC(price_df, [key], start=start_date,
                        end=end_date, lag=lag, data=search_volume)
            corr_by_lag.append(corr)

            if abs(corr) > max_abs_corr:
                max_abs_corr = abs(corr)
                max_lag = lag

        result.append((key, corr_by_lag, max_abs_corr, max_lag))

    # DataFrame으로 만들기
    result_df = pd.DataFrame(result)
    result_df.columns = ['candi_key', 'corr_by_lag', 'max_abs_corr', 'max_lag']
    result_df.sort_values(
        by='max_abs_corr', ascending=False, inplace=True)  # 내림차순 정렬

    return result_df



def get_corr_vol(start_date, end_date):
    '''
    keywords : 검색량 불러올 키워드 리스트 (List)
    start : 주가 시작 날짜 ex. "2016-02-01" (str)
    end : 주가 끝 날짜 ex. "2023-01-26" (str)
    '''
    # final_candi_list 불러오기.
    # 모델이 바뀌면 csv파일을 바꿔줘야 함
    keywords = pd.read_csv(
        "./data/final_candi_list.csv", index_col=0)['keyword']

    # 검색량 불러오기 (dict)
    with open(f"./data/final_candi_search_volume.json", "r") as json_file:
        search_volume = json.load(json_file)

    # 주가 불러오기
    # 코스피에 대한 DataFrame -> columns : ['Date', 'Close']
    price_df = StockPrice(start_date, end_date)

    # correlation 계산하기
    # corr 계산
    result = []
    for key in tqdm(keywords, leave=True):
        corr_by_lag = []
        max_abs_corr = -100  # 임의의 작은 값으로 세팅
        max_lag = -100  # 임의의 값으로 세팅
        for lag in range(1, 11):
            corr = TLCC(price_df, [key], start=start_date,
                        end=end_date, lag=lag, data=search_volume)
            corr_by_lag.append(corr)

            if abs(corr) > max_abs_corr:
                max_abs_corr = abs(corr)
                max_lag = lag

        result.append((key, corr_by_lag, max_abs_corr, max_lag))

    # DataFrame으로 만들기
    result_df = pd.DataFrame(result)
    result_df.columns = ['candi_key', 'corr_by_lag', 'max_abs_corr', 'max_lag']
    result_df.sort_values(
        by='max_abs_corr', ascending=False, inplace=True)  # 내림차순 정렬

    keys = result_df.head(10)['candi_key']
    list = []
    for k in keys:
        k_df = searchVolume2([k], search_volume, start_date,
                             end_date, timeUnit="date").set_index(keys='Date')
        k_df.columns = [k]
        list.append(k_df)
    vol_df = pd.concat(list, axis=1)
    # sklearn minmaxscalar 
    return result_df, vol_df

# def corr_given_time(start, end):
#     time.sleep(5)
#     return pd.DataFrame({'candi_key': ['키워드1', '키워드2', '키워드3', '키워드4', '키워드5'], 'max_abs_corr': [
#         '0.9', '0.9', '0.9', '0.9', '0.9']}, columns=['candi_key', 'max_abs_corr'])
