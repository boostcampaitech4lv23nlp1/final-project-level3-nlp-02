import streamlit as st
import corr_given_time
import datetime
import pandas as pd
import numpy as np
from urllib.error import URLError

set_time = [['2016-02-01', '2017-01-01'], ['2017-01-01', '2018-01-01'], ['2018-01-01', '2019-01-01'], ['2019-01-01',
                                                                                                       '2020-01-01'], ['2020-01-01', '2021-01-01'], ['2021-01-01', '2022-01-01'], ['2022-01-01', '2023-01-01'],]

for i in range(len(set_time)):
    df = corr_given_time.corr_given_time(
        set_time[i][0], set_time[i][1])
    df.head(10).reset_index().to_csv(f"./../data/{2016+i}keyword.csv")
