import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import traceback
from typing import Optional#, Any, Union

date_choice = lambda x, y: y if pd.isna(x) else y
vdate_choice = np.vectorize(date_choice)

@st.cache(allow_output_mutation=True)
def load_data(nrows: Optional[int] = None, cols: Optional[list] = None, errors: str = 'raise'):
    """
    Parameters
    ----------
    nrows : Optional[int], optional
        number of rows to load. The default is None, i.e. all rows.
    cols : Optional[list], optional
        columns to read. The default is None, i.e. all rows.
    errors: str
        what to do with errors (raise, coerce, ignore), as in pd.to_datetime. Default is 'coerce'.
        
    Returns
    -------
    data : pd.DataFrame
        the DataFrame of desired rows and columns, dates are turned to pd.datetime.

    """
    data = pd.read_csv(LINELIST_URL, nrows=nrows, usecols=cols)
    for c in data.columns:
        if c.startswith('Date_'):
            data[c] = pd.to_datetime(data[c], errors=errors) 
    return data 

def ravg_edges(arr: np.ndarray, int_: int):
    """
    Note
    ----
    avoids issues at edges by progressively reducing the size of the average
    interval near beginning and end.
    It ensures, for instance, that ravg_edges(np.ones(N), int_) => np.ones(N)
    Parameters
    ----------
    arr: np.arr
        the array to get the running average of
    int_: int
        the size of the running interval.
    
    Returns
    -------
    np.arr
        the running average
    """
    if not len(arr):
        return np.array([])
    ravg = np.convolve(arr, np.ones(int_)/int_, mode='same')
    offset = int_ // 2
    for o in range(offset):
        ravg[o] = arr[:o+1].mean()
    offset -= 0 if int_%2 == 1 else 1
    for o in range(1, offset+1):
        ravg[-o] = arr[-o:].mean()
    return ravg

def plot_daily(df: pd.DataFrame, key: str = 'only', min_int: int = 3, max_int: int = 15, default: int = 7):
    if 'suspected' in df['Status'].values:
        status = st.selectbox('Cases to consider', ['Only confirmed', 'Confirmed and suspected'])
        filter_ = {'Only confirmed': ['confirmed'], 'Confirmed and suspected': ['confirmed', 'suspected']}[status]
        selected= df[df['Status'].isin(filter_)]
    else:
        selected = df[df['Status'] == 'confirmed']
    if len(selected):
        selected['Date'] = vdate_choice(selected['Date_confirmation'], selected['Date_entry'])
        int_world = st.number_input('Running average interval', min_value=min_int, max_value=max_int, value=default, key=key)
        
        gendered = st.checkbox('Divide by gender', key=f'gendered_{key}')
        try:
            data_to_plot = selected.set_index('Date')['ID'].resample('D').count()
            ravg = ravg_edges(data_to_plot, int_world)
            fig = go.Figure()   
            fig.add_trace(go.Bar(x=data_to_plot.index, y=data_to_plot.values, marker_color='black', name='total cases (T)'))
            fig.add_trace(go.Scatter(x=data_to_plot.index, y=ravg, marker_color='black', name='running average (T)'))
            if gendered:
                for g, colour in zip(['male', 'female'],['blue','pink']):
                    gender = selected[selected['Gender'] == g].set_index('Date')['ID'].resample('D').count()
                    g_ravg = ravg_edges(gender, int_world)
                    fig.add_trace(go.Bar(x=gender.index, y=gender.values, marker_color=colour, name=f'{g} cases ({g[0].upper()})'))
                    fig.add_trace(go.Scatter(x=gender.index, y=g_ravg, marker_color=colour, name=f'running average ({g[0].upper()})'))
            st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.markdown(f'{e}')
            st.text(f'{traceback.format_exc()}')
    else:
        st.markdown('No reported cases match the search criteria.')
            
TITLE = 'Monkey Pox Evolution'
st.set_page_config(page_title=TITLE,
                    page_icon=':chart:',
                    layout='wide',
                    initial_sidebar_state='auto',
                    menu_items={
                                'Get Help': 'https://www.thegraphnetwork.org',
                                'Report a bug': 'https://github.com/thegraphnetwork/MPX-dashboard/issues',
     })
st.sidebar.image('tgn.png')
st.title(TITLE)

LINELIST_URL = 'https://raw.githubusercontent.com/globaldothealth/monkeypox/main/latest.csv'
TS_URL = 'https://raw.githubusercontent.com/globaldothealth/monkeypox/main/timeseries-confirmed.csv'

data_load_state = st.text('Loading data...')
all_cases = load_data(cols=['ID', 'Status', 'Country', 'Gender', 'Date_confirmation', 'Date_entry', 'Symptoms'])
data_load_state.text('Done!')

if st.checkbox('Show all_cases'):
    st.subheader('Linelist data')
    st.dataframe(all_cases)

st.markdown('## Cases globally')
plot_daily(all_cases, key='world')

st.markdown('## Cases by country')
country = st.selectbox('Select country', sorted(list(set(all_cases.Country))))
country_cases = all_cases[all_cases.Country==country]
plot_daily(country_cases, key='country')


    

