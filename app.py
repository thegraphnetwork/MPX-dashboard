import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import traceback
from typing import Optional#, Any, Union

date_choice = lambda x, y: y if pd.isna(x) else y
vdate_choice = np.vectorize(date_choice)
# pd.options.plotting.backend = 'plotly'

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

def get_colours(n):
    if n < 10:
        return px.colors.qualitative.G10
    elif n < 24:
        return px.colors.qualitative.Dark24
    elif n < 26:
        return px.colors.qualitative.Alphabet
    elif n < 60:
        return px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    elif n < 71:
        return px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24 + px.colors.qualitative.Safe
    elif n < 81:
        return px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24 + px.colors.qualitative.Safe + px.colors.qualitative.G10
    
# def plot(df: pd.DataFrame, cumulative: bool = False, entrytype: str = 'cases',
#          key: str = 'only', index_col: str = 'Date', min_int: int = 3,
#          max_int: int = 15, default: int = 7, win_type: Optional[str] = None):
#     """
#     Note
#     ----
#     Uses streamlit user input to further select the data and plots it.
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame with the data to plot
#     cumulative: bool
#         wheter to plot cumulative sum of entries. Default is False.
#         If True, arguments for rolling average are ignored.
#     entrytype: str
#         what the entries are (cases, deaths, ...)
#     key : str, optional
#         Key for streamlit, avoid doubles. The default is 'only'.
#     index_col: str
#         column to use as x axes. Default is date_confirmation if available and date_entry otherwise
#     min_int : int, optional
#         minimum value for rolling average. The default is 3.
#     max_int : int, optional
#         maximum value for rolling average. The default is 15.
#     default : int, optional
#         default value for rolling average number input. The default is 7.
#     win_type : Optional[str], optional
#         window type for rolling average. The default is None.

#     Returns
#     -------
#     None.

#     """
    
#     if 'suspected' in df['Status'].values:
#         status = st.selectbox(f'{entrytype} to consider', ['Only confirmed', 'Confirmed and suspected'])
#         filter_ = {'Only confirmed': ['confirmed'], 'Confirmed and suspected': ['confirmed', 'suspected']}[status]
#         selected= df[df['Status'].isin(filter_)]
#     else:
#         selected = df[df['Status'] == 'confirmed']
#     if len(selected):
#         selected['Date'] = vdate_choice(selected['Date_confirmation'], selected['Date_entry'])
#         gendered = st.checkbox('Divide by gender', key=f'gendered_{key}')
#         try:
#             data_to_plot = selected.set_index(index_col)['ID'].resample('D').count()
#             fig = go.Figure()   
#             if cumulative:
#                 data_to_plot = data_to_plot.cumsum()
#                 fig.add_trace(go.Scatter(x=data_to_plot.index, y=data_to_plot.values, marker_color='black', name=f'cumulative total {entrytype} (T)'))
#             else:
#                 int_ = st.number_input('Running average interval', min_value=min_int, max_value=max_int, value=default, key=key)
#                 ravg = data_to_plot.rolling(int_, win_type=win_type).mean()
#                 fig.add_trace(go.Bar(x=data_to_plot.index, y=data_to_plot.values, marker_color='black', opacity=0.75, name=f'total {entrytype} (T)'))
#                 fig.add_trace(go.Scatter(x=data_to_plot.index, y=ravg, marker_color='black', opacity=0.5, name='running average (T)'))
#             if gendered:
#                 for g, colour in zip(['male', 'female'],['blue','pink']):
#                     gender = selected[selected['Gender'] == g].set_index(index_col)['ID'].resample('D').count()
#                     if cumulative:
#                         fig.add_trace(go.Scatter(x=gender.index, y=gender.values, marker_color=colour, name=f'cumulative {g} {entrytype} ({g[0].upper()})'))
#                     else:
#                         g_ravg = gender.rolling(int_, win_type=win_type).mean()
#                         fig.add_trace(go.Bar(x=gender.index, y=gender.values, marker_color=colour, opacity=0.75, name=f'{g} {entrytype} ({g[0].upper()})'))
#                         fig.add_trace(go.Scatter(x=gender.index, y=g_ravg, marker_color=colour, opacity=0.5, name=f'running average ({g[0].upper()})'))
#             st.plotly_chart(fig, use_container_width=True)
                    
#         except Exception as e:
#             st.markdown(f'{e}')
#             st.text(f'{traceback.format_exc()}')
#     else:
#         st.markdown(f'No reported {entrytype} match the search criteria.')

def plot(df: pd.DataFrame, values: list = [], column: str ='Country',
         cumulative: bool = False, entrytype: str = 'cases',
         daily: bool = True, rolling: bool = True, plot_tot: bool = False,
         tot_label: str = 'World', plot_sumvals: bool = False, 
         sumvals_label: str = 'Sum of selected countries', key: str = 'only',
         index_col: str = 'Date', min_int: int = 3, max_int: int = 15,
         default: int = 7, win_type: Optional[str] = None,
         colours: Optional[list] = None):
    """
    Note
    ----
    Uses streamlit user input to further select the data and plots it.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data to plot
    values : list
        values to consider. For instance, countries to plot as individual lines. Default is [], for only total curve
    column: str
        column for 'values'. Default is 'Country'
    cumulative : bool
        wheter to plot cumulative sum of entries. Default is False.
        If True, arguments for rolling average are ignored.
    entrytype: str
        what the entries are (cases, deaths, ...)
    daily: bool
        whether to plot daily cases. If also rolling, uses bars instead of line
    rolling: bool
        whether to plot rolling average. 
    plot_tot: bool
        whether to plot the values from the whole dataframe (e.g. "World" if column=="Country", both genders if column="Gender")
    tot_label: str
        label for the "total" curve. Default
    plot_sumvals: bool
        whether to plot the sum of the selected values f(e.g. "England + Scotland + Wales")
    sumvals_label: str
        label for the "sum of selected values" curve. Default is "sum of selected countries"
    key : str, optional
        Key for streamlit, avoid doubles. The default is 'only'.
    index_col: str
        column to use as x axes. Default is date_confirmation if available and date_entry otherwise
    min_int : int, optional
        minimum value for rolling average. The default is 3.
    max_int : int, optional
        maximum value for rolling average. The default is 15.
    default : int, optional
        default value for rolling average number input. The default is 7.
    win_type : Optional[str], optional
        window type for rolling average. The default is None.
    colours: list, optional
        list of colours. For instance plotly.express.colors.qualitative.G10.
        If None, combines plotly colours to get enough.

    Returns
    -------
    None.

    """
    values_cases = df if plot_tot else df[df[column].isin(values)]
    
    if 'suspected' in values_cases['Status'].values:
        status = st.selectbox(f'{entrytype} to consider', ['Only confirmed', 'Confirmed and suspected'], key=f'sus_{key}')
        filter_ = {'Only confirmed': ['confirmed'], 'Confirmed and suspected': ['confirmed', 'suspected']}[status]
        selected= values_cases[values_cases['Status'].isin(filter_)]
    else:
        selected = values_cases[values_cases['Status'] == 'confirmed']
    if len(selected):
        if not cumulative and rolling:
            int_ = st.number_input('Running average interval', min_value=min_int, max_value=max_int, value=default, key=key)
        selected['Date'] = vdate_choice(selected['Date_confirmation'], selected['Date_entry'])
        if len(selected[index_col].dropna()):
            fig = go.Figure()  
            plotfunc = go.Bar if daily and rolling else go.Scatter
            if colours:
                if len(colours) < len(values):
                    raise ValueError('You did not provide enough colours for your data!')
            else:
                colours = get_colours(len(values))
            colour_index = 0
            if plot_tot:
                data_to_plot = selected.set_index(index_col)['ID'].resample('D').count()
                if cumulative:
                    data_to_plot = data_to_plot.cumsum()
                    fig.add_trace(go.Scatter(x=data_to_plot.index, y=data_to_plot.values, marker_color=colours[colour_index], name=f'{tot_label}'))
                else:
                    if daily:
                        fig.add_trace(plotfunc(x=data_to_plot.index, y=data_to_plot.values, marker_color=colours[colour_index], opacity=0.75 if daily else 1, name=f'{tot_label}'))
                    if rolling:
                        ravg = data_to_plot.rolling(int_, win_type=win_type).mean()
                        fig.add_trace(go.Scatter(x=data_to_plot.index, y=ravg, marker_color=colours[colour_index], opacity=0.5, name=f'{tot_label}, rolling average'))
                colour_index += 1
            if plot_sumvals:
                vals =  selected[selected[column].isin(values)]
                data_to_plot = vals.set_index(index_col)['ID'].resample('D').count()
                if cumulative:
                    data_to_plot = data_to_plot.cumsum()
                    fig.add_trace(go.Scatter(x=data_to_plot.index, y=data_to_plot.values, marker_color=colours[colour_index], name=f'{sumvals_label}'))
                else:
                    if daily:
                        fig.add_trace(plotfunc(x=data_to_plot.index, y=data_to_plot.values, marker_color=colours[colour_index], opacity=0.75 if daily else 1, name=f'{sumvals_label}'))
                    if rolling:
                        ravg = data_to_plot.rolling(int_, win_type=win_type).mean()
                        fig.add_trace(go.Scatter(x=data_to_plot.index, y=ravg, marker_color=colours[colour_index], opacity=0.5, name=f'{sumvals_label}, rolling average'))
                colour_index += 1
            for n,value in enumerate(values):
                vals = selected[selected[column] == value]
                data_to_plot = vals.set_index(index_col)['ID'].resample('D').count()
                if cumulative:
                    data_to_plot = data_to_plot.cumsum()
                    fig.add_trace(go.Scatter(x=data_to_plot.index, y=data_to_plot.values, marker_color=colours[colour_index + n], name=f'{value}'))
                else:
                    if daily:
                        fig.add_trace(plotfunc(x=data_to_plot.index, y=data_to_plot.values, marker_color=colours[colour_index + n], opacity=0.75 if daily else 1, name=f'{value}'))
                    if rolling:
                        ravg = data_to_plot.rolling(int_, win_type=win_type).mean()
                        if ravg.notna().values.any():  # not to plot only nans
                            fig.add_trace(go.Scatter(x=data_to_plot.index, y=ravg, marker_color=colours[colour_index + n], opacity=0.5, name=f'{value}, rolling average'))
            colour_index = n if values else 0
            if fig['data']:
                fig['data'][0]['showlegend'] = True
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown('Not enough data for rolling average')
        else:
            st.markdown(f'No reported {entrytype} match the search criteria.')
    else:
        st.markdown(f'No reported {entrytype} match the search criteria.')
    
def plot_countries(df, key='countries', **kwargs):
    all_countries = sorted(list(set(df['Country'])))
    sel_countries = st.multiselect('Select countries to compare', all_countries + ['World', 'Sum of selected'], key=f'sel_{key}')
    plot_tot, plot_sumvals = 'World' in sel_countries, 'Sum of selected' in sel_countries
    plot(df, [i for i in sel_countries if i not in ['World', 'Sum of selected']],
                  key=key,plot_sumvals=plot_sumvals, tot_label='World', plot_tot=plot_tot,
                  sumvals_label='Sum of selected countries', **kwargs)

def plot_tot(df, label='', entrytype='cases', key='tot', colour='black', cumulative=False, **kwargs):
    if not label:
        label = f'{"cumulative" if cumulative else "daily"} {entrytype}' 
    plot(df, key=key, plot_tot=True, tot_label=label, colours=[colour], cumulative=cumulative, **kwargs)
    
def plot_genders(df, key='genders', **kwargs):
    plot(df, ['male', 'female'], key=key, column='Gender', plot_tot=True, tot_label='all cases', colours=['black', 'blue', 'pink'], **kwargs)
    
    
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
# all_cases = load_data(cols=['ID', 'Status', 'Country', 'Gender', 'Date_confirmation', 'Date_entry', 'Date_death', 'Symptoms'])
all_cases = load_data(cols=None)
data_load_state.text('Done!')

if st.checkbox('Show all_cases'):
    st.subheader('Linelist data')
    st.dataframe(all_cases)

st.markdown('## Cases globally')
plot_tot(all_cases, entrytype='cases', key='cases_world', cumulative=True)

st.markdown('## Deaths globally')
plot_tot(all_cases[all_cases['Date_death'].notna()], entrytype='deaths', 
            index_col='Date_death', key='deaths_world', cumulative=True)

st.markdown('## Hospitalisations globally')
plot_tot(all_cases[all_cases['Date_hospitalisation'].notna()], entrytype='hospitalisations', 
            index_col='Date_hospitalisation', key='hospitalisations_world', cumulative=True)

st.markdown('## Cases by country')
all_countries = sorted(list(set(all_cases['Country'])))
country = st.selectbox('Select country', all_countries)
country_cases = all_cases[all_cases['Country'] == country]
plot_tot(country_cases, key='cases_country', cumulative=True)

st.markdown(f'### Deaths in {country}')
plot_tot(country_cases[country_cases['Date_death'].notna()], entrytype='deaths',
            index_col='Date_death', key='deaths_country', cumulative=True)

st.markdown('## Compare countries')
st.markdown('## Cases')
plot_countries(all_cases, cumulative=True,  key='countries_cases', daily=False)
st.markdown('## Deaths')
plot_countries(all_cases[all_cases['Date_death'].notna()], cumulative=False,  key='multiple_countries_deaths',
                        daily=False, rolling=True, entrytype='deaths', index_col='Date_death')

    

