import streamlit as st
from codes import corr_given_time
import datetime
import pandas as pd
import numpy as np
from urllib.error import URLError
from sklearn.preprocessing import MinMaxScaler



def get_keywords_with_time():
    scaler = MinMaxScaler()
    st.markdown(f"# 이 키워드 주목! 🔑")
    st.write(
        """
        ### 선택한 기간의 트렌드 키워드
        """
    )

    @st.cache
    def from_data_file(filename):
        url = (
            "http://raw.githubusercontent.com/streamlit/"
            "example-data/master/hello/v1/%s" % filename
        )
        return pd.read_json(url)

    try:
        st.sidebar.markdown("## 기간 지정")

        start_date = st.sidebar.date_input('시작날짜')  # 디폴트로 오늘 날짜가 찍혀 있다.
        end_date = st.sidebar.date_input('종료날짜')  # 디폴트로 오늘 날짜가 찍혀 있다.
        # if st.sidebar.button('from' + (start_date.strftime('%Y-%m-%d')) +
        #                      " to " + (end_date.strftime('%Y-%m-%d'))):
        if end_date-start_date < datetime.timedelta(days=7) or end_date > datetime.date.today():
            st.error("Please select a valid date for a period of 7 days or more")
        else:
            set_time = [start_date, end_date]
            if 'got_df' not in st.session_state:
                st.session_state.got_df = 0
            if 'time' not in st.session_state:
                st.session_state.time = [0, 0]
            if st.session_state.got_df == 0 or st.session_state.time != set_time:
                with st.spinner('Wait for it...'):
                    df, vol_df = corr_given_time.get_corr_vol(
                        set_time[0].strftime("%Y-%m-%d"), set_time[1].strftime("%Y-%m-%d"))
                    st.session_state.time = set_time
                    st.session_state.got_df = 1
                    st.session_state.df = df.head(10).reset_index()
                    vol_df[vol_df.columns] = scaler.fit_transform(vol_df)
                    st.session_state.vol_df = vol_df
                    st.session_state.stockprice = corr_given_time.StockPrice(
                        set_time[0].strftime("%Y-%m-%d"), set_time[1].strftime("%Y-%m-%d")).set_index(keys=['Date'], inplace=False, drop=True)
                    st.session_state.stockprice[['KOSPI']] = scaler.fit_transform(st.session_state.stockprice)
                    
            st.success('Done!')
            df = st.session_state.df[['candi_key',
                                      'max_abs_corr', 'max_lag', 'corr_by_lag']]
            st.dataframe(df)
            st.header("주목 할 키워드")
            keyword_topk = df['candi_key']

            # col_list = st.columns(len(keyword_topk))
            # for i in range(len(keyword_topk)):
            #     if col_list[i].button(keyword_topk[i]):
            #         st.write(
            #             f"KOSPI와 correlation : {df['max_abs_corr'][i]}")

            col_list = st.columns(5)
            for i in range(5):
                if col_list[i].button(keyword_topk[i]):
                    st.write(
                        f"KOSPI와 correlation : {df['max_abs_corr'][i]}")
                if col_list[i].button(keyword_topk[5+i]):
                    st.write(
                        f"KOSPI와 correlation : {df['max_abs_corr'][5+i]}")

            st.subheader("키워드 검색량 추이")
            # st.dataframe(st.session_state.vol_df)
            selected_keyword = st.multiselect('키워드를 선택하세요.', keyword_topk)
            st.line_chart(st.session_state.vol_df[selected_keyword])
            st.subheader("KOSPI 가격 추이")
            # st.dataframe(st.session_state.stockprice['KOSPI'])
            st.line_chart(st.session_state.stockprice['KOSPI'])


        st.write(
            """
        ### 연도별 트렌드 키워드를 확인하세요.
        """
        )

        def print_keys(year):
            # df = pd.read_csv(f"keywords_{year}.csv", index_col=0)
            df = pd.read_csv(f"./data/{year}keyword.csv")
            # df = pd.DataFrame({'candi_key': ['키워드1', '키워드2', '키워드3', '키워드4', '키워드5'], 'corr': [
            #     '0.9', '0.9', '0.9', '0.9', '0.9']}, columns=['candi_key', 'max_abs_corr'])
            col_list2 = st.columns(2)
            with col_list2[0]:
                for i in range(len(df)//2):
                    st.header(df['candi_key'][i*2])
                    st.write(f"KOSPI와 correlation : {df['max_abs_corr'][i*2]}")
            with col_list2[1]:
                for i in range(len(df)//2):
                    st.header(df['candi_key'][i*2+1])
                    st.write(
                        f"KOSPI와 correlation : {df['max_abs_corr'][i*2+1]}")

        keyword_topk = list(range(2016, 2023))

        col_list = st.columns(len(keyword_topk))

        for i in range(len(keyword_topk)):
            if col_list[i].button(str(keyword_topk[i])):
                print_keys(str(keyword_topk[i]))

    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )


get_keywords_with_time()
