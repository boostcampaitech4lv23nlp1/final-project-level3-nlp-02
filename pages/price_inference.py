import streamlit as st
from codes import inference_price
import pandas as pd

st.title("그래서.. 사요? 💸")

df = pd.read_csv("./data/predict_past.csv", index_col=0)

st.subheader("지난 30일 예측")
# df = inference_price.predict_past()
# st.dataframe(df)
st.line_chart(df)

yes_price = df['origin_true_prices'][-1]
# tom_price = inference_price.predict_tomorrow()
file = open("./data/ensemble_tomorrow_price.txt", "r")
tom_price = float(file.readline())
d = round((tom_price-yes_price)/tom_price, 3)
st.header("오늘의 코스피 예측")
st.metric(label="오늘 종가 예측",
          value=f"{round(tom_price,2)} 원", delta=f"{d*100} %")
