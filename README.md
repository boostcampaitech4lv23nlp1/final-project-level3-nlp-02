# ๐ ํผ์คํธํญ๊ท
> Keyword Extraction task๋ฅผ ์ด์ฉํ KOSPI ํค์๋ ์ถ์ถ ๋ฐ KOSPI index ์์ธก

---

## Table of Contents
0. [Archive](https://github.com/boostcampaitech4lv23nlp1/final-project-level3-nlp-02#0-archive)
1. [Team](https://github.com/boostcampaitech4lv23nlp1/final-project-level3-nlp-02#1-team)
2. [Process](https://github.com/boostcampaitech4lv23nlp1/final-project-level3-nlp-02#2-process)
3. [Demo](https://github.com/boostcampaitech4lv23nlp1/final-project-level3-nlp-02#3-demo)
4. [Data](https://github.com/boostcampaitech4lv23nlp1/final-project-level3-nlp-02#4-data)
5. [Model](https://github.com/boostcampaitech4lv23nlp1/final-project-level3-nlp-02#5-model)
6. [How To Use](https://github.com/boostcampaitech4lv23nlp1/final-project-level3-nlp-02#6-how-to-use)

---

## 0. Archive

[๐น ๋ฐํ ์์](https://youtu.be/tZWLKHwTNTU) <br>
[๐ ๋ฐํ ์๋ฃ](https://drive.google.com/file/d/1CscioUS-JbiSrIL7DuPMJpFcQl4-HL3P/view?usp=share_link)

---

## 1. Team

### Members
[๊ณ ์ฐ์ง_T4006]()|[๊น์์ค_T4036]()|[ํ์น์ฝ_T4231]()|
|:-:|:-:|:-:|
|<a href=""><img src="https://user-images.githubusercontent.com/48678866/217627058-ed04a1ab-4cf2-4be3-b2cf-ed83583c57fb.png" width="150" height="200"></a>|<a href=""><img src="https://user-images.githubusercontent.com/48678866/217627098-becef973-2b54-4aa4-8720-77d360818dfa.png" width="150" height="200"></a>|<a href=""><img src="https://user-images.githubusercontent.com/48678866/217627082-aa8f79f7-e580-410a-88bf-ef1b53b000d1.png" width="150" height="200"></a>|

### Contribution

| Member | Contribution | 
| --- | --- |
| ๊ณ ์ฐ์ง(**PM**) | ๋ผ๋ฌธ ์กฐ์ฌ, Embedding model ๊ตฌํ ๋ฐ ํ์ต, Price prediction model ๊ตฌํ ๋ฐ ํ์ต |
| ๊น์์ค | ๋ฐ์ดํฐ ๊ตฌ์ถ ๋ฐ ์ฒ๋ฆฌ, Embedding model ๊ตฌํ ๋ฐ ํ์ต, Demo ์ ์, Batch serving ๊ตฌ์ถ |
| ํ์น์ฝ | ๋ผ๋ฌธ ์กฐ์ฌ, ๋ฐ์ดํฐ EDA, ๊ฒ์๋ ๋ฐ์ดํฐ ์์ง, Embedding model ๊ตฌํ ๋ฐ ํ์ต |

---
<br>

## 2. Process

<p align="center">
<img src="https://user-images.githubusercontent.com/48678866/217632326-0513fea5-b3af-488a-9d1f-7b14599ef8ae.png" width="800" height="500">
</p>
  
---
<br>

## 3. Data

> **Namuwiki Text** : huggingface์ ์๋ก๋๋์ด ์๋ [๋คํํ์ผ](https://huggingface.co/datasets/heegyu/namuwiki-extracted) ์ด์ฉ

> **Seed keyword** : [ํต๊ณ์ฒญ ์ ๊ณต ๊ฒฝ์ ํค์๋](https://data.kostat.go.kr/social/keyword/index.do), ๋ผ๋ฌธ, ๊ตฌ๊ธ๋ง์ ํตํด KOSPI์ ์ฐ๊ด์ฑ ๋์ ํค์๋ ์ง์ 

> **๋ค์ด๋ฒ ๊ฒ์๋** : [๋ค์ด๋ฒ Developers](https://developers.naver.com/main/) ๋ฐ์ดํฐ๋ฉ API ์ด์ฉํ์ฌ ์์ง

> **KOSPI index** : ์ผํ ํ์ด๋ธ์ค์์ ์ ๊ณตํ๋ KOSPI(์ฝ๋ : ^KS11) - [yfinance](https://github.com/ranaroussi/yfinance) ๋ผ์ด๋ธ๋ฌ๋ฆฌ ํ์ฉํ์ฌ ์์ง
---
<br>

## 4. Model
  
### For Text Embedding
  
#### KLUE RoBERTa large ([Link](https://huggingface.co/klue/roberta-large))

> RoBERTa ๋ชจ๋ธ์ ํ๊ตญ์ด ๋ฐ์ดํฐ(KLUE)๋ฅผ ์ด์ฉํด pre-trainingํ ์ธ์ด ๋ชจ๋ธ
  
#### KPF-BERT ([Link](https://github.com/KPFBERT/kpfbert))

> ํ๊ตญ์ธ๋ก ์งํฅ์ฌ๋จ์์ ๊ตฌ์ถํ 20๋์น์ ๋ฌํ๋ ์ฝ 4์ฒ๋ง ๊ฑด์ ๋ด์ค๊ธฐ์ฌ ๋ฐ์ดํฐ๋ฅผ ์ด์ฉํด ํ์ตํ ๋ชจ๋ธ
  
#### KB-ALBERT ([Link](https://github.com/KB-AI-Research/KB-ALBERT))

> ๊ตฌ๊ธ์ ALBERT์ ๊ฒฝ์ /๊ธ์ต ๋๋ฉ์ธ์ ํนํ๋ ๋๋์ ํ๊ตญ์ด ๋ฐ์ดํฐ๋ฅผ ํ์ต์ํจ ๋ชจ๋ธ

<br>

### For Predicting KOSPI index

#### LSTM

<img src="https://user-images.githubusercontent.com/48678866/217641005-14463045-93c1-4e30-bd64-dcdb25e20bb3.png" width="40%">

---
<br>

## 5. Demo
### ์๋น์ค ๊ตฌ์กฐ
<img src="https://user-images.githubusercontent.com/66728415/217638602-f9836b42-db91-477b-b708-6dd006d0d5b2.png" width="70%">

### ๐ฅ๏ธ Web ์์(Streamlit)

<img src="https://user-images.githubusercontent.com/66728415/217637499-b8743dd9-fdff-4c96-a00f-8f36fbb2df95.gif" width="70%">
<img src="https://user-images.githubusercontent.com/66728415/217637712-a6a22806-cdc3-42cf-a1bf-d7f50b72f27f.gif" width="70%">


<br>

## 6. How to Use

### File Directory

```bash
โโโ codes
โ   โโโ corr_given_time.py
โ   โโโ get_anual.py
โ   โโโ inference_price.py
โโโ dags
โ   โโโ operator_dag.py
โโโ data
โ   โโโ 2016keyword.csv
โ   โโโ 2017keyword.csv
โ   โโโ 2018keyword.csv
โ   โโโ 2019keyword.csv
โ   โโโ 2020keyword.csv
โ   โโโ 2021keyword.csv
โ   โโโ 2022keyword.csv
โ   โโโ ensemble_tomorrow_price.txt
โ   โโโ final_candi_list.csv
โ   โโโ final_candi_search_volume.json
โ   โโโ predict_past.csv
โโโ pages
โ   โโโ get_keywords.py
โ   โโโ price_inference.py
โโโ .gitignore
โโโ README.md
โโโ main.py
โโโ requirements.txt
```

### ๊ฐ์ํ๊ฒฝ 

```
# ๊ฐ์ํ๊ฒฝ ์์ฑ
python3 -m venv $ENV_NAME
# ๊ฐ์ํ๊ฒฝ ํ์ฑํ
source $ENV_NAME/bin/activate
# ๋ผ์ด๋ธ๋ฌ๋ฆฌ ์ค์น
pip3 install --upgrade pip
pip3 install -r requirements.txt
# ๊ฐ์ํ๊ฒฝ ์ข๋ฃ
deactivate
```

### Streamlit
```
streamlit run main.py
```

### Airflow
```
# ์ ๋๊ฒฝ๋ก๋ก ๊ธฐ๋ณธ ๋๋ ํ ๋ฆฌ ์ง์ 
export AIRFLOW_HOME=~/nlp02
# airflow DB ์ด๊ธฐํ -> ๊ธฐ๋ณธ ํ์ผ ์์ฑ
airflow db init
airflow users create --username admin --password 1234 --firstname boocam --lastname kim --role Admin --email xxx@naver.com
airflow webserver --port 8080

# ์ค์ผ์ค๋ฌ ์คํ
export AIRFLOW_HOME=~/nlp02
airflow scheduler
```
---
