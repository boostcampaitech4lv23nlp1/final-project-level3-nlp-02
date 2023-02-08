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
|<img src="https://user-images.githubusercontent.com/48678866/217627058-ed04a1ab-4cf2-4be3-b2cf-ed83583c57fb.png" width="150" height="200">|<img src="https://user-images.githubusercontent.com/48678866/217627098-becef973-2b54-4aa4-8720-77d360818dfa.png" width="150" height="200">|<img src="https://user-images.githubusercontent.com/48678866/217627082-aa8f79f7-e580-410a-88bf-ef1b53b000d1.png" width="150" height="200">|

### Contribution

| Member | Contribution | 
| --- | --- |
| 고우진(**PM**) | 논문 조사, Embedding model 구현 및 학습, Price prediction model 구현 및 학습 |
| 김상윤 | 데이터 구축 및 처리, Embedding model 구현 및 학습, Demo 제작, Batch serving 구축 |
| 현승엽 | 논문 조사, 데이터 EDA, 검색량 데이터 수집, Embedding model 구현 및 학습 |

---

## 2. Process

---

## 3. Demo

--- 

## 4. Data

> **Namuwiki Text** : huggingface에 업로드되어 있는 [덤프파일](https://huggingface.co/datasets/heegyu/namuwiki-extracted) 이용<br>
> **Seed keyword** : [통계청 제공 경제키워드](https://data.kostat.go.kr/social/keyword/index.do), 논문, 구글링을 통해 KOSPI와 연관성 높은 키워드 지정<br>
> **네이버 검색량** : [네이버 Developers](https://developers.naver.com/main/) 데이터랩 API 이용하여 수집 <br>
> **KOSPI index** : 야후 파이낸스에서 제공하는 KOSPI(코드 : ^KS11) - [yfinance](https://github.com/ranaroussi/yfinance) 라이브러리 활용하여 수집
---

## 5. Model

---

## 6. How to Use

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
## File Directory

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
