from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from datetime import datetime, timedelta
import urllib.request
import json
from collections import deque
import torch.nn as nn
import yfinance as yf
from datetime import datetime, timedelta
import torch
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from transformers import is_torch_available
import numpy as np
import matplotlib.pyplot as plt


import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))



default_args = {
    'owner': 'sykim',
    # 이전 DAG의 Task가 성공, 실패 여부에 따라 현재 DAG 실행 여부가 결정. False는 과거의 실행 결과 상관없이 매일 실행한다
    'depends_on_past': False,
    'start_date': datetime(2023, 2, 1),
    'retries': 1,  # 실패시 재시도 횟수
    'retry_delay': timedelta(hours=1)  # 만약 실패하면 5분 뒤 재실행
}

def ex():
    print("heeloworld")

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

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


class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x, y

def load_data1(stock, look_back, num_test):
    # 참고로 stock은 dataframe 형태이어야 한다(series는 안된다)(y부분 추후에 고치기)
    # 현재 argument이름을 stock으로 했지만, scaled_df라고 하는 것이 조금 더 나을 듯 -> 나중에 고치자!
    data_raw = stock.values  # convert to numpy array
    data = []

    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back + 1])

    data = np.array(data)
    test_set_size = int(num_test)
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-1, :]
    # y_train = data[:train_set_size,-1,:]
    y_train = data[:train_set_size, -1, 0].reshape(-1, 1)

    x_valid = data[train_set_size:, :-1]
    # y_valid = data[train_set_size:,-1,:]
    y_valid = data[train_set_size:, -1, 0].reshape(-1, 1)

    return [x_train, y_train, x_valid, y_valid]

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

# function to create train, test data given stock data and sequence length
# train_lstm_cpi.py에 있는 것과는 다르게 train_x, train_y만 return 한다.
def load_data2(stock, look_back):
    # 참고로 stock은 dataframe 형태이어야 한다(series는 안된다)(y부분 추후에 고치기)
    # 현재 argument이름을 stock으로 했지만, scaled_df라고 하는 것이 조금 더 나을 듯 -> 나중에 고치자!
    data_raw = stock.values  # convert to numpy array
    data = []

    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back + 1])

    data = np.array(data)
    train_set_size = data.shape[0]

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, 0].reshape(-1, 1)

    return [x_train, y_train]


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

def model_data_func(price_df, corr_df, topk, start, end, data):
    '''
    price_df: 코스피가격 불러와서 적당히 처리하고 저장한 dataframe(Date가 str임에 주의)
    corr_df: (dataframe)
    topk: 몇개의 candi_key를 lstm의 input으로 사용할 것인지(int)
    start: 시작일(str)
    end: 끝일(str)
    data: volume_dict(dict)

    return: (dataframe) columns=["close","var1",...,"vartopk"]
    '''

    output = pd.DataFrame()

    # 1. price_df -> start~end까지의 stock_name의 가격데이터 가져오기
    # column = (Date, Close)
    output["close"] = price_df["Close"]

    # 2. search_volume_df -> start-(lag-1)~end-(lag-1)까지의 key_name의 검색량 데이터 가져오기
    # column = (Date, Search Volume)

    for i in range(topk):
        key_name = corr_df.iloc[i]["candi_key"]
        lag = int(corr_df.iloc[i]["max_lag"])
        search_volume_start = str(pd.Timestamp(
            start).date() - timedelta(lag-1))
        search_volume_df = searchVolume2(
            [key_name], data, search_volume_start, end)

        # price_df의 Date에서 lag-1 만큼 뺀뒤, 해당 값들을 str으로 만들어서 time이라는 리스트에 집어 넣자
        time = list(map(str, (price_df["Date"]-timedelta(lag-1)).values))

        # search_volume_df에서 Date가 time에 해당하는 행만 뽑아서 search_volume_lagged_df 만들기!
        search_volume_lagged_df = search_volume_df[search_volume_df["Date"].isin(
            time)].reset_index(drop=True)

        output[f"var{i+1}"] = search_volume_lagged_df["Search Volume"]

    return output

# 과거 past일 만큼의 가격을 예측하고 그려주는 함수
def predict_past(past=30):

    print(is_torch_available())
    print(torch.cuda.is_available())

    # seed 설정
    # set_seed(777)

    years = [1, 2, 3, 4, 5, 6, 7]

    # 모델1~7y 별 하이퍼파라미터 정의
    hyperparameter = {1: {"num_epochs": 49, "batch_size": 32, "lr": 0.1, "topk": 0, "look_back": 7, "num_layers": 1, "hidden_dim": 32},
                      2: {"num_epochs": 32, "batch_size": 32, "lr": 0.01, "topk": 3, "look_back": 7, "num_layers": 2, "hidden_dim": 32},
                      3: {"num_epochs": 38, "batch_size": 32, "lr": 0.1, "topk": 0, "look_back": 30, "num_layers": 1, "hidden_dim": 32},
                      4: {"num_epochs": 46, "batch_size": 32, "lr": 0.01, "topk": 3, "look_back": 14, "num_layers": 2, "hidden_dim": 32},
                      5: {"num_epochs": 36, "batch_size": 32, "lr": 0.01, "topk": 5, "look_back": 30, "num_layers": 2, "hidden_dim": 16},
                      6: {"num_epochs": 40, "batch_size": 32, "lr": 0.01, "topk": 1, "look_back": 14, "num_layers": 1, "hidden_dim": 16},
                      7: {"num_epochs": 35, "batch_size": 32, "lr": 0.01, "topk": 1, "look_back": 14, "num_layers": 2, "hidden_dim": 32}
                      }

    # 각각의 1~7y 모델 train 및 inference
    temp = []
    for year in years:
        # 변수세팅
        stock_name = "^KS11"
        # end = datetime.today().strftime("%Y-%m-%d")
        end = "2023-01-20"
        print(1)
        start = str(pd.Timestamp(end).date() - timedelta(365*year))
        print(2)
        print(f"start: {start}, end: {end}")

        num_epochs = hyperparameter[year]["num_epochs"]
        batch_size = hyperparameter[year]["batch_size"]
        lr = hyperparameter[year]["lr"]
        topk = hyperparameter[year]["topk"]
        look_back = hyperparameter[year]["look_back"]

        input_dim = topk+1  # 종가+topk개의 변수
        hidden_dim = hyperparameter[year]["hidden_dim"]
        num_layers = hyperparameter[year]["num_layers"]
        output_dim = 1

        # test_ratio = 0 #엄밀히는 valid_ratio 이다 #x_train, y_train, x_valid, y_valid 만들기에서

        # price_df 만들기 -> StockPrice 함수랑 비교해서 동일하면 대체하기
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
        price_df = df.drop(["Open", "High", "Low", "Adj Close",
                           "Volume", "Dividends", "Stock Splits"], axis=1)
        print("price_df 완성!")
        print(price_df)
        print(f"price_df의 길이: {len(price_df)}")

        # volume_dict 만들기 <- 이게 오래걸린다!
        with open(f"./data/final_candi_search_volume.json", "r") as json_file:
            volume_dict = json.load(json_file)
        print("volume_dict 완성!")

        # corr_df 만들기
        corr_df = corr_given_time.corr_given_time(start, end)
        print(corr_df.columns)
        print("corr_df 완성!")
        print(corr_df)

        # model_data_df 만들기
        model_data_df = model_data_func(
            price_df=price_df, corr_df=corr_df, topk=topk, start=start, end=end, data=volume_dict)
        print(model_data_df)

        # scaled_model_data_df 만들기
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_model_data_df = scaler.fit_transform(
            model_data_df.values).transpose()  # array이다
        scaled_model_data_df = pd.DataFrame(
            dict(zip(list(model_data_df.columns), scaled_model_data_df)))  # 열이름 붙이기

        # x_train, y_train 만들기
        x_train, y_train, x_valid, y_valid = load_data1(
            scaled_model_data_df, look_back, past)
        x_train = torch.Tensor(x_train)  # tensor로 바꾸기
        x_valid = torch.Tensor(x_valid)  # tensor로 바꾸기
        y_train = torch.Tensor(y_train)  # tensor로 바꾸기
        y_valid = torch.Tensor(y_valid)  # tensor로 바꾸기

        print('x_train.shape = ', x_train.shape)
        print('y_train.shape = ', y_train.shape)
        print('x_valid.shape = ', x_valid.shape)
        print('y_valid.shape = ', y_valid.shape)

        # train_dataset, train_dataloader만들기
        train_dataset = CustomDataset(x_train, y_train)
        valid_dataset = CustomDataset(x_valid, y_valid)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        # 이걸 False로 해야 순서대로 예측값을 표현할 수 있음!
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False)

        print('x_train.shape = ', x_train.shape)
        print('y_train.shape = ', y_train.shape)
        print('x_valid.shape = ', x_valid.shape)
        print('y_valid.shape = ', y_valid.shape)

        # setting Device
        device = "cpu"
        print(f"Using {device} device")

        # model 만들기
        model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim,
                     output_dim=output_dim, num_layers=num_layers)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # lr도 인자로 빼내자

        # Train
        for epoch in tqdm(range(num_epochs)):
            # Train mode -----------------------------------------------------------------
            model.train()
            train_loss = 0
            for data, target in train_dataloader:
                optimizer.zero_grad()  # <- pytorch specific operation

                # breakpoint()
                output = model(data)  # 여기에 .to(device) 붙여도 여전히 에러가 뜬다!c
                # Loss calculation
                loss_func = torch.nn.MSELoss()
                loss = loss_func(output, target)
                loss.backward()
                optimizer.step()      # <- parameter update 수행
                train_loss += loss.item()*len(output)

            train_loss /= len(train_dataloader.dataset)
            print("\n[Train] Epoch: {}, MSE loss: {:.4f}".format(
                epoch, train_loss))

        # Inference future past days
        true_prices = []
        pred_prices = []

        model.eval()
        with torch.no_grad():
            for data, target in valid_dataloader:
                pred = model(data)  # 내예상 (batch_size, 1)인 텐서 형태

                # (batch_size,) 형태의 np.array
                target = target.reshape(-1).detach().numpy()
                # (batch_size,) 형태의 np.array
                pred = pred.reshape(-1).detach().numpy()

                true_prices.append(target)
                pred_prices.append(pred)

        true_prices = np.concatenate(true_prices)  # (past,) 형태의 np.array
        pred_prices = np.concatenate(pred_prices)  # (past,) 형태의 np.array

        # 원래대로 scale_back 하기
        true_prices_df = pd.DataFrame(
            dict(zip(range(topk+1), [true_prices for _ in range(topk+1)])))
        pred_prices_df = pd.DataFrame(
            dict(zip(range(topk+1), [pred_prices for _ in range(topk+1)])))

        origin_true_prices = scaler.inverse_transform(
            true_prices_df.values).transpose()[0]
        origin_pred_prices = scaler.inverse_transform(
            pred_prices_df.values).transpose()[0]
        # (past, ) 형태의 np.array
        print(f"origin_pred_prices의 shape: {origin_pred_prices.shape}")

        # temp에 가격 추가하기
        temp.append(origin_pred_prices)

        # 그래프 그리기 -> 시간 되면 x축 실제날짜로 구현되게 하기!!!
        # Visualising the results
        figure, axes = plt.subplots(figsize=(15, 6))
        axes.xaxis_date()
        axes.plot(price_df["Date"][-past:], origin_true_prices,
                  color='red', label=f'Real KOSPI Price of past {past}days')
        axes.plot(price_df["Date"][-past:], origin_pred_prices,
                  color='blue', label=f'Predicted KOSPI Price of past {past}days')
        # axes.xticks(np.arange(0,394,50))
        plt.title(f'KOSPI Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('KOSPI Price')
        plt.legend()
        plt.savefig(f'KOSPI Price of past {past}days_by_model_{year}.png')
        # plt.show()

    temp = np.array(temp)  # (7, past) 형태의 np.array(모델이 7개 이니까!)
    ensemble_pred_prices = temp.mean(axis=0)  # (past,) 형태의 np.array

    # 과거 past일 동안의 (날짜, 실제가격, 앙상블한 예측가격)
    df_return = pd.DataFrame({'origin_true_prices': origin_true_prices,
                             'ensemble_pred_prices': ensemble_pred_prices}, index=price_df["Date"][-past:])
    # 함수를 수행할 dags 디렉토리를 기준으로 상대경로 시작.
    df_return.to_csv("./data/predict_past.csv")
    return df_return
    # return price_df["Date"][-past:], origin_true_prices, ensemble_pred_prices


def predict_tomorrow():

    print(is_torch_available())
    print(torch.cuda.is_available())

    # seed 설정
    # set_seed(777)

    years = [1, 2, 3, 4, 5, 6, 7]

    # 모델1~7y 별 하이퍼파라미터 정의
    hyperparameter = {1: {"num_epochs": 49, "batch_size": 32, "lr": 0.1, "topk": 0, "look_back": 7, "num_layers": 1, "hidden_dim": 32},
                      2: {"num_epochs": 32, "batch_size": 32, "lr": 0.01, "topk": 3, "look_back": 7, "num_layers": 2, "hidden_dim": 32},
                      3: {"num_epochs": 38, "batch_size": 32, "lr": 0.1, "topk": 0, "look_back": 30, "num_layers": 1, "hidden_dim": 32},
                      4: {"num_epochs": 46, "batch_size": 32, "lr": 0.01, "topk": 3, "look_back": 14, "num_layers": 2, "hidden_dim": 32},
                      5: {"num_epochs": 36, "batch_size": 32, "lr": 0.01, "topk": 5, "look_back": 30, "num_layers": 2, "hidden_dim": 16},
                      6: {"num_epochs": 40, "batch_size": 32, "lr": 0.01, "topk": 1, "look_back": 14, "num_layers": 1, "hidden_dim": 16},
                      7: {"num_epochs": 35, "batch_size": 32, "lr": 0.01, "topk": 1, "look_back": 14, "num_layers": 2, "hidden_dim": 32}
                      }

    temp = []
    # 각각의 1~7y 모델 train 및 inference
    for year in years:
        # 변수세팅
        stock_name = "^KS11"
        # end = datetime.today().strftime("%Y-%m-%d")
        end = "2023-01-20"
        print(1)
        start = str(pd.Timestamp(end).date() - timedelta(365*year))
        print(2)
        print(f"start: {start}, end: {end}")

        num_epochs = hyperparameter[year]["num_epochs"]
        batch_size = hyperparameter[year]["batch_size"]
        lr = hyperparameter[year]["lr"]
        topk = hyperparameter[year]["topk"]
        look_back = hyperparameter[year]["look_back"]

        input_dim = topk+1  # 종가+topk개의 변수
        hidden_dim = hyperparameter[year]["hidden_dim"]
        num_layers = hyperparameter[year]["num_layers"]
        output_dim = 1

        # test_ratio = 0 #엄밀히는 valid_ratio 이다 #x_train, y_train, x_valid, y_valid 만들기에서

        # price_df 만들기 -> StockPrice 함수랑 비교해서 동일하면 대체하기
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
        price_df = df.drop(["Open", "High", "Low", "Adj Close",
                           "Volume", "Dividends", "Stock Splits"], axis=1)
        print("price_df 완성!")

        # volume_dict 만들기 <- 이게 오래걸린다!
        with open(f"./data/final_candi_search_volume.json", "r") as json_file:
            volume_dict = json.load(json_file)
        print("volume_dict 완성!")

        # corr_df 만들기
        corr_df = corr_given_time.corr_given_time(start, end)
        print(corr_df.columns)
        print("corr_df 완성!")

        # model_data_df 만들기
        model_data_df = model_data_func(
            price_df=price_df, corr_df=corr_df, topk=topk, start=start, end=end, data=volume_dict)

        # scaled_model_data_df 만들기
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_model_data_df = scaler.fit_transform(
            model_data_df.values).transpose()  # array이다
        scaled_model_data_df = pd.DataFrame(
            dict(zip(list(model_data_df.columns), scaled_model_data_df)))  # 열이름 붙이기

        # x_train, y_train 만들기
        x_train, y_train = load_data2(scaled_model_data_df, look_back)
        x_train = torch.Tensor(x_train)  # tensor로 바꾸기
        y_train = torch.Tensor(y_train)  # tensor로 바꾸기

        print('x_train.shape = ', x_train.shape)
        print('y_train.shape = ', y_train.shape)

        # train_dataset, train_dataloader만들기
        train_dataset = CustomDataset(x_train, y_train)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)

        # setting Device
        device = "cpu"
        print(f"Using {device} device")

        # model 만들기
        model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim,
                     output_dim=output_dim, num_layers=num_layers)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # lr도 인자로 빼내자

        # Train
        for epoch in tqdm(range(num_epochs)):
            # Train mode -----------------------------------------------------------------
            model.train()
            train_loss = 0
            for data, target in train_dataloader:
                optimizer.zero_grad()  # <- pytorch specific operation

                # breakpoint()
                output = model(data)  # 여기에 .to(device) 붙여도 여전히 에러가 뜬다!c
                # Loss calculation
                loss_func = torch.nn.MSELoss()
                loss = loss_func(output, target)
                loss.backward()
                optimizer.step()      # <- parameter update 수행
                train_loss += loss.item()*len(output)

            train_loss /= len(train_dataloader.dataset)
            print("\n[Train] Epoch: {}, MSE loss: {:.4f}".format(
                epoch, train_loss))

        # Inference tomorrow
        pred_prices = []
        # 학습에 쓰지 않았던 것 부터 첫 eval 대상으로 삼기!
        x = scaled_model_data_df.iloc[-look_back:]
        x = x.values  # (look_back, topk+1), np.array

        model.eval()
        with torch.no_grad():
            # 텐서로 바꾸고 (1, look_back, topk+1) 형태로 바꾸기
            inference_input = torch.Tensor(
                x[-look_back:].reshape(1, look_back, topk+1))
            pred = model(inference_input)  # (batch_size=1, 1)
            pred_prices.append(pred.item())

        # 원래대로 scale_back 하기
        pred_prices_df = pd.DataFrame(
            dict(zip(range(topk+1), [pred_prices for _ in range(topk+1)])))
        print(f"pred_prices_df의 shape: {pred_prices_df.shape}")
        origin_pred_prices = scaler.inverse_transform(
            pred_prices_df.values).transpose()[0]
        print(f"origin_pred_prices의 shape: {origin_pred_prices.shape}")

        print(f"model{year}가 예측하는 내일의 코스피 가격은 {origin_pred_prices} 입니다!!!")

        temp.append(origin_pred_prices)

    temp = np.array(temp)
    print(temp)
    ensemble_tomorrow_price = temp.mean()
    print(f"앙상블 내일 가격 {ensemble_tomorrow_price}")
    # 함수를 수행할 dags 디렉토리를 기준으로 상대경로 시작.
    f = open("./data/ensemble_tomorrow_price.txt", 'w')
    f.write(f'{ensemble_tomorrow_price}')
    f.close()
    return ensemble_tomorrow_price








# with 구문으로 DAG 정의
with DAG(
        dag_id='get_vol_train_inference',
        default_args=default_args,
        # UTC 시간 기준 12시 00분에 Daily로 실행하겠다! 
        schedule_interval='00 12 * * *',
        # schedule_interval='@once',
        tags=['my_dags']
) as dag:
    t1 = PythonOperator(
        task_id='get_vol',
        python_callable=get_search_volume  # 실행할 python 함수
        # python_callable=ex
    )

    t2 = PythonOperator(
        task_id='predict_past',
        python_callable=predict_past  # 실행할 python 함수
        # python_callable=ex
    )

    t3 = PythonOperator(
        task_id='predict_tomorrow',
        python_callable=predict_tomorrow  # 실행할 python 함수
        # python_callable=ex
    )

    t1 >> t2
    t1 >> t3
