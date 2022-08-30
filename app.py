import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import traceback


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

pd.options.plotting.backend = 'plotly'

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

DATE_COLUMN = 'date/time'
LINELIST_URL = 'https://raw.githubusercontent.com/globaldothealth/monkeypox/main/latest.csv'
TS_URL = 'https://raw.githubusercontent.com/globaldothealth/monkeypox/main/timeseries-confirmed.csv'
# """
# As far as I understand you guys are pulling updated databases all the time. 
# I'd like to learn more about how you do it
# """

@st.cache(allow_output_mutation=True)
def load_data(nrows=None):
    ll_data = pd.read_csv(LINELIST_URL, nrows=nrows)
    # ts_data = pd.read_csv(TS_URL, nrows=nrows)
    # lowercase = lambda x: str(x).lower()
    # data.rename(lowercase, axis='columns', inplace=True)
    # data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return ll_data #, ts_data

data_load_state = st.text('Loading data...')
# linelist, timeseries = load_data()
all_cases = load_data()
data_load_state.text('Done!')

for c in all_cases.columns:
    if c.startswith('Date_'):
        all_cases[c] = pd.to_datetime(all_cases[c], errors='coerce')
        
#TODO
# """
# This part of code is a bit tricky for me. All dates are already datetimes in the current file.
# Furthermore, coercing can be risky. It puts full trust in our starting file. Why not using 'ignore'?
# In case of weird entries they would appear. Why not a lambda function which converts any non-NaT, 
# and raising errors for any non-NaT that cannot be converted to a date?
# """

if st.checkbox('Show all_cases'):
    st.subheader('Linelist data')
    st.dataframe(all_cases)

st.markdown('## Cases globally')

if 'suspected' in all_cases['Status'].values:
    status = st.selectbox('Cases to consider', ['Only confirmed', 'Confirmed and suspected'])
    filter_ = {'Only confirmed': ['confirmed'], 'Confirmed and suspected': ['confirmed', 'suspected']}[status]
    selected_cases= all_cases[all_cases['Status'].isin(filter_)]
else:
    selected_cases = all_cases[all_cases['Status'] == 'confirmed']

date_choice = lambda x, y: y if pd.isna(x) else y
vdate_choice = np.vectorize(date_choice)
if len(selected_cases):
    selected_cases['Date'] = vdate_choice(selected_cases['Date_confirmation'], selected_cases['Date_entry'])
    int_world = st.number_input('Running average interval', min_value=3, max_value=15, value=7, key='int_world')
    
    gendered = st.checkbox('Divide by gender', key='gendered_world')
    try:
        data_to_plot = selected_cases.set_index('Date')['ID'].resample('D').count()
        ravg = ravg_edges(data_to_plot, int_world)
        fig = go.Figure()   
        fig.add_trace(go.Bar(x=data_to_plot.index, y=data_to_plot.values, marker_color='black', name='total cases (T)'))
        fig.add_trace(go.Scatter(x=data_to_plot.index, y=ravg, marker_color='black', name='running average (T)'))
        if gendered:
            for g, colour in zip(['male', 'female'],['blue','pink']):
                gender = selected_cases[selected_cases['Gender'] == g].set_index('Date')['ID'].resample('D').count()
                g_ravg = ravg_edges(gender, int_world)
                fig.add_trace(go.Bar(x=gender.index, y=gender.values, marker_color=colour, name='{} cases ({})'.format(g, g[0].upper())))
                fig.add_trace(go.Scatter(x=gender.index, y=g_ravg, marker_color=colour, name='running average ({})'.format(g[0].upper())))
        st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.markdown('{}'.format(e))
        st.text('{}'.format(traceback.format_exc()))

st.markdown('## Cases by country')

country = st.selectbox('Select country', sorted(list(set(all_cases.Country))))
country_list = all_cases[all_cases.Country==country]

if 'suspected' in country_list['Status'].values:
    status = st.selectbox('Cases to consider', ['Only confirmed', 'Confirmed and suspected'])
    filter_ = {'Only confirmed': ['confirmed'], 'Confirmed and suspected': ['confirmed', 'suspected']}[status]
    country_list= country_list[country_list['Status'].isin(filter_)]
else:
    country_list= country_list[country_list['Status'] == 'confirmed']

date_choice = lambda x, y: y if pd.isna(x) else y
vdate_choice = np.vectorize(date_choice)
if len(country_list):
    country_list['Date'] = vdate_choice(country_list['Date_confirmation'], country_list['Date_entry'])
    int_country = st.number_input('Running average interval', min_value=3, max_value=15, value=7, key='int_country')
    
    gendered = st.checkbox('Divide by gender', key='gendered_country')
    try:
        data_to_plot = country_list.set_index('Date')['ID'].resample('D').count()
        ravg = ravg_edges(data_to_plot, int_country)
        fig = go.Figure()   
        fig.add_trace(go.Bar(x=data_to_plot.index, y=data_to_plot.values, marker_color='black', name='total cases (T)'))
        fig.add_trace(go.Scatter(x=data_to_plot.index, y=ravg, marker_color='black', name='running average (T)'))
        if gendered:
            for g, colour in zip(['male', 'female'],['blue','pink']):
                gender = country_list[country_list['Gender'] == g].set_index('Date')['ID'].resample('D').count()
                g_ravg = ravg_edges(gender, int_country)
                fig.add_trace(go.Bar(x=gender.index, y=gender.values, marker_color=colour, name='{} cases ({})'.format(g, g[0].upper())))
                fig.add_trace(go.Scatter(x=gender.index, y=g_ravg, marker_color=colour, name='running average ({})'.format(g[0].upper())))
        st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.markdown('{}'.format(e))
        st.text('{}'.format(traceback.format_exc()))
else:
    st.markdown('No reported cases match the search criteria.')
    

