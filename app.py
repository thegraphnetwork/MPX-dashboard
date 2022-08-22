import streamlit as st
import pandas as pd
import numpy as np

pd.options.plotting.backend = "plotly"

TITLE = 'Monkey Pox Evolution'
st.set_page_config(page_title=TITLE,
                    page_icon=":chart:",
                    layout="wide",
                    initial_sidebar_state="auto",
                    menu_items={
                                'Get Help': 'https://www.thegraphnetwork.org',
                                'Report a bug': "https://github.com/thegraphnetwork/MPX-dashboard/issues",
     })
st.sidebar.image('tgn.png')
st.title(TITLE)

DATE_COLUMN = 'date/time'
LINELIST_URL = 'https://raw.githubusercontent.com/globaldothealth/monkeypox/main/latest.csv'
TS_URL = 'https://raw.githubusercontent.com/globaldothealth/monkeypox/main/timeseries-confirmed.csv'

@st.cache(allow_output_mutation=True)
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

for c in linelist.columns:
    if c.startswith('Date_'):
        linelist[c] = pd.to_datetime(linelist[c], errors='coerce')


if st.checkbox('Show linelist table'):
    st.subheader('Linelist data')
    st.dataframe(linelist)

country = st.selectbox('Select country', list(set(linelist.Country)))
country_list = linelist[linelist.Country==country]
try:
    fig = country_list.set_index('Date_confirmation')['ID'].resample('W').count().plot()
    st.plotly_chart(fig, use_container_width=True)
except:
    st.markdown('No cases reported so far.')
