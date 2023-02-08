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

---

## 5. Model

---

## 6. How to Use

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
