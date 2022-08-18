import streamlit as st
import pandas as pd
import numpy as np


st.title('Monkey Pox Evolution')

DATE_COLUMN = 'date/time'
LINELIST_URL = 'https://raw.githubusercontent.com/globaldothealth/monkeypox/main/latest.csv'
TS_URL = 'https://raw.githubusercontent.com/globaldothealth/monkeypox/main/timeseries-confirmed.csv'

@st.cache
def load_data(nrows):
    ll_data = pd.read_csv(LINELIST_URL, nrows=nrows)
    ts_data = pd.read_csv(TS_URL, nrows=nrows)
    # lowercase = lambda x: str(x).lower()
    # data.rename(lowercase, axis='columns', inplace=True)
    # data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return ll_data, ts_data

data_load_state = st.text('Loading data...')
linelist, timeseries = load_data(10000)
data_load_state.text("Done!")

if st.checkbox('Show linelist table'):
    st.subheader('Linelist data')
    st.dataframe(linelist)


