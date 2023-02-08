import json
import torch.nn as nn
import yfinance as yf
from datetime import datetime, timedelta
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from transformers import is_torch_available
import numpy as np
import matplotlib.pyplot as plt

from codes import corr_given_time


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

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_dim).requires_grad_()  # x.size(0)은 배치사이즈 인듯
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_() #x.size(0)은 배치사이즈 인듯
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_dim).requires_grad_()
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> , (N,L,D*H_out) N:배치사이즈, L:seq_length, D:1or2(우리는1), H_out:hidden dim
        # out[:, -1, :] --> (N,L) --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> (N,1) 인지 (N,) 인지 조금 헷갈림
        return out


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


# function to create train, test data given stock data and sequence length
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
        search_volume_df = corr_given_time.searchVolume2(
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
