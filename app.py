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

def plot_daily(df: pd.DataFrame, entrytype: str = "cases", key: str = 'only', min_int: int = 3,
               max_int: int = 15, default: int = 7, win_type: Optional[str] = None):
    """
    Note
    ----
    Uses streamlit user input to further select the data and plots it.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data to plot
    entrytype: str
        what the entries are (cases, deaths, ...)
    key : str, optional
        Key for streamlit, avoid doubles. The default is 'only'.
    min_int : int, optional
        minimum value for rolling average. The default is 3.
    max_int : int, optional
        maximum value for rolling average. The default is 15.
    default : int, optional
        default value for rolling average number input. The default is 7.
    win_type : Optional[str], optional
        window type for rolling average. The default is None.

    Returns
    -------
    None.

    """
    
    if 'suspected' in df['Status'].values:
        status = st.selectbox(f'{entrytype} to consider', ['Only confirmed', 'Confirmed and suspected'])
        filter_ = {'Only confirmed': ['confirmed'], 'Confirmed and suspected': ['confirmed', 'suspected']}[status]
        selected= df[df['Status'].isin(filter_)]
    else:
        selected = df[df['Status'] == 'confirmed']
    if len(selected):
        selected['Date'] = vdate_choice(selected['Date_confirmation'], selected['Date_entry'])
        int_ = st.number_input('Running average interval', min_value=min_int, max_value=max_int, value=default, key=key)
        
        gendered = st.checkbox('Divide by gender', key=f'gendered_{key}')
        try:
            data_to_plot = selected.set_index('Date')['ID'].resample('D').count()
            ravg = data_to_plot.rolling(int_, win_type=win_type).mean()
            fig = go.Figure()   
            fig.add_trace(go.Bar(x=data_to_plot.index, y=data_to_plot.values, marker_color='black', name=f'total {entrytype} (T)'))
            fig.add_trace(go.Scatter(x=data_to_plot.index, y=ravg, marker_color='black', name='running average (T)'))
            if gendered:
                for g, colour in zip(['male', 'female'],['blue','pink']):
                    gender = selected[selected['Gender'] == g].set_index('Date')['ID'].resample('D').count()
                    g_ravg = gender.rolling(int_, win_type=win_type).mean()
                    fig.add_trace(go.Bar(x=gender.index, y=gender.values, marker_color=colour, name=f'{g} {entrytype} ({g[0].upper()})'))
                    fig.add_trace(go.Scatter(x=gender.index, y=g_ravg, marker_color=colour, name=f'running average ({g[0].upper()})'))
            st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.markdown(f'{e}')
            st.text(f'{traceback.format_exc()}')
    else:
        st.markdown(f'No reported {entrytype} match the search criteria.')
            
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
# all_cases = load_data(cols=['ID', 'Status', 'Country', 'Gender', 'Date_confirmation', 'Date_entry', 'Symptoms'])
all_cases = load_data(cols=None)
data_load_state.text('Done!')

if st.checkbox('Show all_cases'):
    st.subheader('Linelist data')
    st.dataframe(all_cases)

st.markdown('## Cases globally')
plot_daily(all_cases, entrytype="cases", key='cases_world', win_type='exponential')
st.markdown('## Deaths globally')
plot_daily(all_cases[all_cases["Date_death"].notna()], entrytype="deaths", key='deaths_world', win_type='exponential')

st.markdown('## Cases by country')
country = st.selectbox('Select country', sorted(list(set(all_cases.Country))))
country_cases = all_cases[all_cases.Country==country]
plot_daily(country_cases, key='cases_country', win_type='exponential')
st.markdown(f'### Deaths in {country}')
plot_daily(country_cases[country_cases["Date_death"].notna()], entrytype="deaths", key='deaths_country', win_type='exponential')


    

