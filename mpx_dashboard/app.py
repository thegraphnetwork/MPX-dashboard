import streamlit as st
from mpx_dashboard import csv_specs
import os
# import pandas as pd
from mpx_dashboard.plotting import total_weekly_metrics, plot_tot, plot_countries, \
    evolution_on_map, barstack_countries
from mpx_dashboard.read_write import cached_read_csv, cached_read_parquet
    
TITLE = 'Global Monkey Pox Evolution'
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

st.sidebar.markdown("Powered by [EpiGraphHub](https://epigraphhub.org)")
rawfile = 'https://raw.githubusercontent.com/globaldothealth/monkeypox/main/latest_deprecated.csv'

data_load_state = st.text('Loading data...')
folder = '../data'
cases = cached_read_parquet(os.path.join(folder,'cases.parquet'))
sus_cases = cached_read_parquet(os.path.join(folder,'sus_cases.parquet'))
deaths = cached_read_parquet(os.path.join(folder,'deaths.parquet'))
sus_deaths = cached_read_parquet(os.path.join(folder,'sus_deaths.parquet'))
data_load_state.text('Read aggregated data!')


all_countries = sorted(list(set(cases[csv_specs.countrycol]).union(set(sus_cases[csv_specs.countrycol]))))
population = cached_read_csv(os.path.join(folder,'population.csv'),index_col=0).iloc[:,0]
pop_load_state = st.text('Read population!')

world_tab, country_tab, comparison_tab = st.tabs(['Global', 'Country', 'Compare'])
with world_tab:
    st_columns = st.columns(4)
    do_sus = False
    if len(sus_cases):
        status = st_columns[0].selectbox('Entries to consider', ['Only confirmed', 'Confirmed and suspected'], key='sus_selector_world')
        do_sus = True if status == 'Confirmed and suspected' else False
    total_weekly_metrics(cases, aggr_sus=sus_cases, key='cases', do_sus=do_sus)
    total_weekly_metrics(deaths, aggr_sus=sus_deaths, key='deaths', do_sus=do_sus)    
    st.markdown('## Cases globally')
    plot_tot(cases, aggr_sus=sus_cases, key='cases_world', do_sus=do_sus, st_columns=None)
    
    st.markdown('## Deaths globally')
    plot_tot(deaths, sus_deaths, key='deaths_world', do_sus=do_sus, st_columns=None)
    
with country_tab:
    st.markdown('## Cases and deaths by country')
    st_columns = st.columns(4)
    country = st_columns[0].selectbox('Select country', all_countries)
    country_cases = cases[cases[csv_specs.countrycol] == country]
    country_deaths = deaths[deaths[csv_specs.countrycol] == country]
    country_sus_cases = sus_cases[sus_cases[csv_specs.countrycol] == country]
    do_sus = False
    i_col = 1
    if len(country_sus_cases):
        status = st_columns[1].selectbox('Entries to consider', ['Only confirmed', 'Confirmed and suspected'], key='sus_selector_country')
        do_sus = True if status == 'Confirmed and suspected' else False
        i_col += 1
    scale_country = st_columns[i_col].checkbox('Scale by population', key='scale_country')
    country_sus_deaths = sus_deaths[sus_deaths[csv_specs.countrycol] == country]
    if scale_country:
        country_cases_scaled, country_sus_cases_scaled = country_cases.copy(), country_sus_cases.copy()
        country_deaths_scaled, country_sus_deaths_scaled = country_deaths.copy(), country_sus_deaths.copy()
        country_cases_scaled['daily_cases'] = country_cases_scaled['daily_cases']/country_cases_scaled[csv_specs.countrycol].map(population).astype(float)
        country_sus_cases_scaled['daily_cases'] = country_sus_cases_scaled['daily_cases']/country_sus_cases_scaled[csv_specs.countrycol].map(population).astype(float)
        country_deaths_scaled['daily_deaths'] = country_deaths_scaled['daily_deaths']/country_deaths_scaled[csv_specs.countrycol].map(population).astype(float)
        country_sus_deaths_scaled['daily_deaths'] = country_sus_deaths_scaled['daily_deaths']/country_sus_deaths_scaled[csv_specs.countrycol].map(population).astype(float)
    total_weekly_metrics(country_cases, aggr_sus=country_sus_cases, do_sus=do_sus)
    total_weekly_metrics(country_deaths, aggr_sus=country_sus_deaths, do_sus=do_sus)   
    
    st.markdown(f'### Cases in {country}')
    plot_tot(country_cases_scaled if scale_country else country_cases,
             aggr_sus=country_sus_cases_scaled if scale_country else country_sus_cases,
             key='cases_country', st_columns=None, do_sus=do_sus)
    
    st.markdown(f'### Deaths in {country}')
    plot_tot(country_deaths_scaled if scale_country else country_deaths,
             aggr_sus=country_sus_deaths_scaled if scale_country else country_sus_deaths,
             key='deaths_country', st_columns=None, do_sus=do_sus)

with comparison_tab:
    st.markdown('## Compare countries')
    st_columns = st.columns(3)
    scale = st_columns[0].checkbox('Scale by population', key='scale_compare')
    if scale:
        cases_scaled, sus_cases_scaled = cases.copy(), sus_cases.copy()
        deaths_scaled, sus_deaths_scaled = deaths.copy(), sus_deaths.copy()
        cases_scaled['daily_cases'] = cases_scaled['daily_cases']/cases_scaled[csv_specs.countrycol].map(population).astype(float)
        sus_cases_scaled['daily_cases'] = sus_cases_scaled['daily_cases']/sus_cases_scaled[csv_specs.countrycol].map(population).astype(float)
        deaths_scaled['daily_deaths'] = deaths_scaled['daily_deaths']/deaths_scaled[csv_specs.countrycol].map(population).astype(float)
        sus_deaths_scaled['daily_deaths'] = sus_deaths_scaled['daily_deaths']/sus_deaths_scaled[csv_specs.countrycol].map(population).astype(float)
    st.markdown('### Cases')
    plot_countries(cases_scaled if scale else cases,
                   aggr_sus=sus_cases_scaled if scale else sus_cases,
                   key='countries_cases')
    st.markdown('### Deaths')
    plot_countries(deaths_scaled if scale else deaths,
                   aggr_sus=sus_deaths_scaled if scale else sus_deaths,
                   key='countries_deaths')
    
    st.markdown('## Barstacked')
    st.markdown('### Cases')
    barstack_countries(cases_scaled if scale else cases,
                   aggr_sus=sus_cases_scaled if scale else sus_cases,
                       cumulative=True,  key='barstack_countries')
    
    st.markdown('## Maps')
    st.markdown('### Cases')
    evolution_on_map(cases_scaled if scale else cases,
                    aggr_sus=sus_cases_scaled if scale else sus_cases,
                      key='cases_map')
    
    




