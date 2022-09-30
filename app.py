import streamlit as st
import csv_specs
import os
import pandas as pd
from functions import load_cases, group_and_aggr, total_weekly_metrics, plot_tot, plot_countries, \
    evolution_on_map, barstack_countries, add_to_parquet
    
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
with st.sidebar.expander('Additional Information'):
    st.markdown('Data from [Global.Health](https://github.com/globaldothealth/monkeypox "https://github.com/globaldothealth/monkeypox")')
    st.markdown('Rolling averages use an exponential weighing of the data.')

# casesfile = 'https://raw.githubusercontent.com/globaldothealth/monkeypox/main/latest.csv'
casesfile = 'data_20Sept2022.csv'
# TS_URL = 'https://raw.githubusercontent.com/globaldothealth/monkeypox/main/timeseries-confirmed.csv'
# cols_needed = ['ID', 'Status', 'Country', 'Country_ISO3', 'Date_confirmation', 'Date_entry', 'Date_death']
cols_needed = None

data_load_state = st.text('Loading data...')
if os.path.isfile('lines_read.txt'):
    with open('lines_read.txt', 'r') as f:
        skiprows = int(f.read())
else:
    skiprows = 0
new_cases = load_cases(casesfile, usecols=None, skiprows=skiprows)
data_load_state.text('Data loaded!')

newrows = len(new_cases)
with open('lines_read.txt', 'w') as f:
    f.write(f'{skiprows + newrows}')

if newrows:
    new_aggr_cases = group_and_aggr(new_cases, column=csv_specs.countrycol, date_col=csv_specs.confdatecol,
                        dropna=True, entrytype='cases', dropzeros=True)
    add_to_parquet(new_aggr_cases, 'cases.parquet')
    new_aggr_deaths = group_and_aggr(new_cases, column=csv_specs.countrycol, date_col=csv_specs.deathdatecol,
                        dropna=True, entrytype='deaths', dropzeros=True)
    add_to_parquet(new_aggr_deaths, 'deaths.parquet')

cases = pd.read_parquet('cases.parquet', columns=cols_needed)
deaths = pd.read_parquet('deaths.parquet', columns=cols_needed)
# if st.checkbox('Show all_cases'):
#     st.subheader('Linelist data')
#     st.dataframe(all_cases)

world_tab, country_tab, comparison_tab = st.tabs(['Global', 'Country', 'Compare'])
with world_tab:
    total_weekly_metrics(all_cases, entrytype='cases', index_col='Date_confirmation')
    total_weekly_metrics(all_cases[all_cases['Date_death'].notna()], entrytype='deaths', index_col='Date_death')    
    st.markdown('## Cases globally')
    plot_tot(all_cases, entrytype='cases', key='cases_world')
    
    st.markdown('## Deaths globally')
    plot_tot(all_cases[all_cases['Date_death'].notna()], entrytype='deaths', 
                index_col='Date_death', key='deaths_world')
    
    # st.markdown('## Hospitalisations globally')
    # plot_tot(all_cases[all_cases['Date_hospitalisation'].notna()], entrytype='hospitalisations', 
    #             index_col='Date_hospitalisation', key='hospitalisations_world', cumulative=False)

with country_tab:
    st.markdown('## Cases by country')
    all_countries = sorted(list(set(all_cases['Country'])))
    st_columns = st.columns(4)
    country = st_columns[0].selectbox('Select country', all_countries)
    country_cases = all_cases[all_cases['Country'] == country]
    
    total_weekly_metrics(country_cases, entrytype='cases', index_col='Date_confirmation')
    total_weekly_metrics(country_cases[country_cases['Date_death'].notna()], entrytype='deaths', index_col='Date_death')    
    
    st.markdown(f'### Cases in {country}')
    plot_tot(country_cases, key='cases_country', st_columns=st_columns[-3:])
    
    st.markdown(f'### Deaths in {country}')
    plot_tot(country_cases[country_cases['Date_death'].notna()], entrytype='deaths',
                index_col='Date_death', key='deaths_country')

with comparison_tab:
    st.markdown('## Compare countries')
    st.markdown('### Cases')
    plot_countries(all_cases,  key='countries_cases')
    st.markdown('### Deaths')
    plot_countries(all_cases[all_cases['Date_death'].notna()],  key='multiple_countries_deaths',
                            entrytype='deaths', index_col='Date_death')
    
    st.markdown('## Barstacked')
    st.markdown('### Cases')
    barstack_countries(all_cases, cumulative=True,  key='barstack_countries')
    
    st.markdown('## Maps')
    st.markdown('### Cases')
    evolution_on_map(all_cases, key='cases_map')
    
    




