# 📈 퍼스트펭귄
> Keyword Extraction task를 이용한 KOSPI 키워드 추출 및 KOSPI index 예측

---

## Table of Contents
0. Archive
1. Introduction
2. Process
3. Demo
4. Data
5. Model
6. How To Use

---

## 0. Archive

---

## 1. Introduction

### Members
[고우진_T4006]()|[김상윤_T4036]()|[현승엽_T4231]()|
|:-:|:-:|:-:|
|<a href=""><img src="https://user-images.githubusercontent.com/48678866/217627058-ed04a1ab-4cf2-4be3-b2cf-ed83583c57fb.png" width="150" height="200"></a>|<a href=""><img src="https://user-images.githubusercontent.com/48678866/217627098-becef973-2b54-4aa4-8720-77d360818dfa.png" width="150" height="200"></a>|<a href=""><img src="https://user-images.githubusercontent.com/48678866/217627082-aa8f79f7-e580-410a-88bf-ef1b53b000d1.png" width="150" height="200"></a>|

### Contribution

| Member | Contribution | 
| --- | --- |
| 고우진(**PM**) | 논문 조사, Embedding model 구현 및 학습, Price prediction model 구현 및 학습 |
| 김상윤 | 데이터 구축 및 처리, Embedding model 구현 및 학습, Demo 제작, Batch serving 구축 |
| 현승엽 | 논문 조사, 데이터 EDA, 검색량 데이터 수집, Embedding model 구현 및 학습 |

---
<br>

## 2. Process

<p align="center">
<img src="https://user-images.githubusercontent.com/48678866/217632326-0513fea5-b3af-488a-9d1f-7b14599ef8ae.png" width="800" height="500">
</p>
  
---
<br>

## 3. Data

> **Namuwiki Text** : huggingface에 업로드되어 있는 [덤프파일](https://huggingface.co/datasets/heegyu/namuwiki-extracted) 이용

> **Seed keyword** : [통계청 제공 경제키워드](https://data.kostat.go.kr/social/keyword/index.do), 논문, 구글링을 통해 KOSPI와 연관성 높은 키워드 지정

> **네이버 검색량** : [네이버 Developers](https://developers.naver.com/main/) 데이터랩 API 이용하여 수집

> **KOSPI index** : 야후 파이낸스에서 제공하는 KOSPI(코드 : ^KS11) - [yfinance](https://github.com/ranaroussi/yfinance) 라이브러리 활용하여 수집
---
<br>

## 4. Model
  
### For Text Embedding
  
#### KLUE RoBERTa large ([Link](https://huggingface.co/klue/roberta-large))

> RoBERTa 모델을 한국어 데이터(KLUE)를 이용해 pre-training한 언어 모델
  
#### KPF-BERT ([Link](https://github.com/KPFBERT/kpfbert))

> 한국언론진흥재단에서 구축한 20년치에 달하는 약 4천만 건의 뉴스기사 데이터를 이용해 학습한 모델
  
#### KB-ALBERT ([Link](https://github.com/KB-AI-Research/KB-ALBERT))

> 구글의 ALBERT에 경제/금융 도메인에 특화된 대량의 한국어 데이터를 학습시킨 모델

<br>

### For Predicting KOSPI index

#### LSTM


---
<br>

## 5. Demo
### 서비스 구조
<img src="https://user-images.githubusercontent.com/66728415/217638602-f9836b42-db91-477b-b708-6dd006d0d5b2.png" width="70%">

### 🖥️ Web 예시(Streamlit)

<img src="https://user-images.githubusercontent.com/66728415/217637499-b8743dd9-fdff-4c96-a00f-8f36fbb2df95.gif" width="70%">
<img src="https://user-images.githubusercontent.com/66728415/217637712-a6a22806-cdc3-42cf-a1bf-d7f50b72f27f.gif" width="70%">


<br>

## 6. How to Use

### File Directory

```bash
├── codes
│   ├── corr_given_time.py
│   ├── get_anual.py
│   └── inference_price.py
├── dags
│   └── operator_dag.py
├── data
│   ├── 2016keyword.csv
│   ├── 2017keyword.csv
│   ├── 2018keyword.csv
│   ├── 2019keyword.csv
│   ├── 2020keyword.csv
│   ├── 2021keyword.csv
│   ├── 2022keyword.csv
│   ├── ensemble_tomorrow_price.txt
│   ├── final_candi_list.csv
│   ├── final_candi_search_volume.json
│   └── predict_past.csv
├── pages
│   ├── get_keywords.py
│   └── price_inference.py
├── .gitignore
├── README.md
├── main.py
└── requirements.txt
```

### 가상환경 

```
# 가상환경 생성
python3 -m venv $ENV_NAME
# 가상환경 활성화
source $ENV_NAME/bin/activate
# 라이브러리 설치
pip3 install --upgrade pip
pip3 install -r requirements.txt
# 가상환경 종료
deactivate
```

### Streamlit
```
streamlit run main.py
```

### Airflow
```
# 절대경로로 기본 디렉토리 지정
export AIRFLOW_HOME=~/nlp02
# airflow DB 초기화 -> 기본 파일 생성
airflow db init
airflow users create --username admin --password 1234 --firstname boocam --lastname kim --role Admin --email xxx@naver.com
airflow webserver --port 8080

# 스케줄러 실행
export AIRFLOW_HOME=~/nlp02
airflow scheduler
```
---
