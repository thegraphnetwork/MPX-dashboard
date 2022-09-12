#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 19:10:04 2022

@author: nico
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional#, Any, Union

date_choice = lambda x, y: y if pd.isna(x) else y
vdate_choice = np.vectorize(date_choice)

@st.cache(allow_output_mutation=True)
def load_data(file: str, nrows: Optional[int] = None, cols: Optional[list] = None, errors: str = 'raise'):
    """
    Parameters
    ----------
    file: str
        the csv to read
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
    data = pd.read_csv(file, nrows=nrows, usecols=cols)
    for c in data.columns:
        if c.startswith('Date_'):
            data[c] = pd.to_datetime(data[c], errors=errors) 
    return data 

@st.cache()
def get_daily_count(df: pd.DataFrame, index_col: str, count_col: str = 'ID'):
    """
    Parameters
    ----------
    df : pd.DataFrame
        dataframe with a case per row
    index_col : str
        date column to use as index
    count_col : str
        column to use to count the cases. Default is 'ID'

    Returns
    -------
    pd.Series
        DESCRIPTION.

    """
    return df.set_index(index_col)[count_col].resample('D').count()

def na_percentage(df: pd.DataFrame):
    """
    Parameters
    ----------
    df: pd.DataFrame
        the dataframe to analyse
    
    Returns
    -------
    pd.DataFrame
        contains the percentage of nan per column, omitting when 0%
    """
    count_na = lambda df, colname: len(df[df[colname].isna()])
    tot = len(df)/100
    dict_ = {}
    for col in df.columns:
        na_count = count_na(df, col)
        if na_count:
            dict_[col] =  na_count/tot
    return pd.DataFrame(pd.Series(dict_, name='percentage_nans'))

def total_weekly_metrics(df: pd.DataFrame, entrytype: str = 'cases',
                         index_col: str = 'Date_confirmation', count_col: str = 'ID'):
    """
    Parameters
    ----------
    df : pd.DataFrame
        dataframe with one entrytype per line
    entrytype : str, optional
       cases/deaths/hospitalisations. The default is 'cases'.
   index_col : str, optional
       date column to use as index. The default is 'Date_confirmation'.
   count_col : str, optional
       column to count. The default is 'ID'.

    Returns
    -------
    None. Displays metrics in two columns

    """
    daily_count = get_daily_count(df, index_col=index_col, count_col=count_col)
    last_week = daily_count.iloc[-7:].sum()
    previous_week = daily_count.iloc[-14:-7].sum()  # Risky because of delay in reporting
    cols = st.columns(2)
    cols[0].metric(label=f'Total {entrytype}', value=daily_count.sum())
    cols[1].metric(label=f'{entrytype.capitalize()} in the last 7 days', value=last_week, delta=f'{last_week-previous_week}')  # convert to string because delta does not accept np.int64
            
def get_colours(n: int):
    """
    Parameters
    ----------
    n : int
        the number of desired colours.

    Returns
    -------
    list
        list of at least n colours.

    """
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

def plot(df: pd.DataFrame, values: list = [], column: str ='Country',
         cumulative: Optional[bool] = None, entrytype: str = 'cases',
         daily: Optional[bool] = None, rolling: Optional[bool] = None,
         plot_tot: bool = False, tot_label: str = 'World',
         plot_sumvals: bool = False, sumvals_label: str = 'Sum of selected countries',
         key: str = 'only', index_col: str = 'Date', min_int: int = 3,
         max_int: int = 15, default: int = 7, win_type: Optional[str] = None,
         colours: Optional[list] = None, st_columns: list = []):
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
    cumulative : bool, optional
        wheter to plot cumulative sum of entries. Default is None. In such case, a user input is used.
        If True, arguments for rolling average are ignored.
    entrytype: str
        what the entries are (cases, deaths, ...)
    daily: bool, optional
        whether to plot daily cases. If also rolling, uses bars instead of line. Default is None. In such case, a user input is used.
    rolling: bool, optional
        whether to plot rolling average. Default is None. In such case, a user input is used.
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
    st_columns: list, optional
        list of streamlit columns. If [], creates 2 columns.
        Provide empty columns, or all columns, if others are already used.
        
    Returns
    -------
    None.

    """
    if not st_columns:
        st_columns = st.columns(3)
    i_col = -3
    values_cases = df if plot_tot else df[df[column].isin(values)]
    
    if 'suspected' in values_cases['Status'].values:
        with st_columns[i_col]:
            status = st.selectbox(f'{entrytype} to consider', ['Only confirmed', 'Confirmed and suspected'], key=f'sus_{key}')
        i_col += 1
        filter_ = {'Only confirmed': ['confirmed'], 'Confirmed and suspected': ['confirmed', 'suspected']}[status]
        selected= values_cases[values_cases['Status'].isin(filter_)]
    else:
        selected = values_cases[values_cases['Status'] == 'confirmed']
    if len(selected):
        if (daily, rolling) == (None, None):
                if cumulative == None:
                    cumulative = st_columns[i_col].checkbox('Cumulative', key=f'cumul_{key}')
                daily = st_columns[i_col].checkbox('Daily', key=f'daily_{key}',
                                                   value=True if not cumulative else False,
                                                   disabled=True if cumulative else False)
                i_col += 1
                rolling = st_columns[i_col].checkbox('Rolling average',
                                                     key=f'rolling_{key}',
                                                     disabled=True if cumulative else False)
        if not cumulative and rolling:
            with st_columns[i_col]:
                int_ = st.number_input('Rolling average interval', min_value=min_int, max_value=max_int, value=default, key=key)
        selected['Date'] = vdate_choice(selected['Date_confirmation'], selected['Date_entry'])
        if len(selected[index_col].dropna()):
            fig = go.Figure()  
            plotfunc = go.Bar if daily and rolling else go.Scatter
            ncurves = len(values) + 1 if plot_tot else 0 + 1 if plot_sumvals else 0
            if colours:
                if len(colours) < ncurves:
                    raise ValueError('You did not provide enough colours for your data!')
            else:
                colours = get_colours(ncurves)
            colour_index = 0
            if plot_tot:
                data_to_plot = get_daily_count(selected, index_col)
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
                data_to_plot = get_daily_count(vals, index_col)
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
                if pd.isna(value):
                    vals = selected[selected[column].isna()]
                else:
                    vals = selected[selected[column] == value]
                data_to_plot = get_daily_count(vals, index_col)
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
            if fig['data']:
                fig['data'][0]['showlegend'] = True
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown('Not enough data for rolling average')
        else:
            st.markdown(f'No reported {entrytype} match the search criteria.')
    else:
        st.markdown(f'No reported {entrytype} match the search criteria.')
    
def plot_countries(df: pd.DataFrame, key: str = 'countries',
                   cumulative: Optional[bool] = None, daily: Optional[bool] = None,
                   rolling: Optional[bool] = None, **kwargs):
    """
    Note
    ----
    Plots countries selected by the user

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to start from.
    key : str
        A key for streamlit. The default is 'countries'.
    cumulative : bool, optional
        wheter to plot cumulative sum of entries. Default is None. In such case, a user input is used.
        If True, arguments for rolling average are ignored.
    entrytype: str
        what the entries are (cases, deaths, ...)
    daily: bool, optional
        whether to plot daily cases. If also rolling, uses bars instead of line. Default is None. In such case, a user input is used.
    rolling: bool, optional
        whether to plot rolling average. Default is None. In such case, a user input is used.
    **kwargs : 
        Any other keyword argument for the generic plot function.

    Returns
    -------
    None.

    """
    all_countries = sorted(list(set(df['Country'])))
    st_columns = st.columns(4)
    with st_columns[0]:
        sel_countries = st.multiselect('Select countries to compare', all_countries + ['World', 'Sum of selected'], key=f'sel_{key}')
    plot_tot, plot_sumvals = 'World' in sel_countries, 'Sum of selected' in sel_countries
    plot(df, values=[i for i in sel_countries if i not in ['World', 'Sum of selected']],
                  key=key,plot_sumvals=plot_sumvals, tot_label='World', plot_tot=plot_tot,
                  sumvals_label='Sum of selected countries', st_columns=st_columns[-3:],
                  cumulative=cumulative, daily=daily, rolling=rolling, **kwargs)

def plot_tot(df: pd.DataFrame, label: str = '', entrytype: str = 'cases', key: str = 'tot',
             colour: str = 'black', cumulative: Optional[bool] = None, 
             daily: Optional[bool] = None, rolling: Optional[bool] = None, **kwargs):
    
    """
    Note
    ----
    plots the whole dataframe according to the selection criteria

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to start from.
    label : str
        The label for your single curve. The default is "daily"/"cumulative" entrytype.
    entrytype : str
        The type of entry. The default is 'cases'. Other possibilities include deaths, hospitalisations
    key : str, optional
        Key for streamlit. The default is 'tot'.
    colour : str, optional
        Colour for the curve. The default is 'black'.
    cumulative : bool, optional
        wheter to plot cumulative sum of entries. Default is None. In such case, a user input is used.
        If True, arguments for rolling average are ignored.
    daily: bool, optional
        whether to plot daily cases. If also rolling, uses bars instead of line. Default is None. In such case, a user input is used.
    rolling: bool, optional
        whether to plot rolling average. Default is None. In such case, a user input is used.
    **kwargs : 
        Any other keyword argument for the generic plot function.

    Returns
    -------
    None.

    """
    if not label:
        label = f'{"cumulative" if cumulative else "daily"} {entrytype}' 
    plot(df, key=key, plot_tot=True, tot_label=label, colours=[colour],
         entrytype=entrytype, cumulative=cumulative, daily=daily, rolling=rolling, **kwargs)
    
def plot_genders(df: pd.DataFrame, key: str = 'genders', cumulative: Optional[bool] = None,
                 daily: Optional[bool] = None, rolling: Optional[bool] = None, **kwargs):
    """
    Note
    ----
    Plots curves for each gender

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe to start from.
    key : str, optional
        A key for streamlir. The default is 'genders'.
    cumulative : bool, optional
        wheter to plot cumulative sum of entries. Default is None. In such case, a user input is used.
        If True, arguments for rolling average are ignored.
    daily: bool, optional
        whether to plot daily cases. If also rolling, uses bars instead of line. Default is None. In such case, a user input is used.
    rolling: bool, optional
        whether to plot rolling average. Default is None. In such case, a user input is used.
    **kwargs : 
        Any other keyword argument for the generic plot function.

    Returns
    -------
    None.

    """
    plot(df, values=['male', 'female'], key=key, column='Gender', plot_tot=True,
         cumulative=cumulative, daily=daily, rolling=rolling, tot_label='all cases',
         colours=['black', 'blue', 'pink'], **kwargs)
    
def plot_outcome(df: pd.DataFrame, include_nan: bool = True, key: str = 'outcome', **kwargs):
    """
    Note
    ----
    Plots the outcome of cases

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to start from.
    include_nan : bool
        Whether to plot the nans. The default is True.
    key : str
        Key for streanlit. The default is 'outcome'.
    **kwargs : 
        Any other keyword argument for the generic plot function.

    Returns
    -------
    None.

    """
    values = ["Death", "Recovered", np.nan] if include_nan else ["Death", "Recovered"]
    plot(df, values=values, column='Outcome', cumulative=True, **kwargs)

def plot_needed_hospital(df: pd.DataFrame, include_nan: bool = True, key: str = 'hospitalised', **kwargs):
    """
    Note
    ----
    Plots how many cases needed hospitalisation

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to start from.
    include_nan : bool
        Whether to plot the nans. The default is True.
    key : str
        Key for streanlit. The default is 'outcome'.
    **kwargs : 
        Any other keyword argument for the generic plot function.

    Returns
    -------
    None.

    """
    values = ["Y", "N", np.nan] if include_nan else ["Y", "N"]
    plot(df, values=values, column='Outcome', cumulative=True, **kwargs)

def barstack(df: pd.DataFrame, values: list = [], column: str ='Country',
         cumulative: Optional[bool] = None, entrytype: str = 'cases',
         daily: bool = True, key: str = 'barstack',
         index_col: str = 'Date',
         colours: Optional[list] = None, st_columns: list = []):
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
    cumulative : bool, Optional
        wheter to plot cumulative sum of entries. Default is None, and asks for user input.
        If True, arguments for rolling average are ignored.
    entrytype: str
        what the entries are (cases, deaths, ...)
    key : str, optional
        Key for streamlit, avoid doubles. The default is 'only'.
    index_col: str
        column to use as x axes. Default is date_confirmation if available and date_entry otherwise
    colours: list, optional
        list of colours. For instance plotly.express.colors.qualitative.G10.
        If None, combines plotly colours to get enough.
    st_columns: list, optional
        list of streamlit columns. If [], creates 2 columns.
        Provide empty columns, or all columns, if others are already used.
    
    Returns
    -------
    None.

    """
    if not st_columns:
        st_columns = st.columns(2)
    i_col = -2
    values_cases = df if plot_tot else df[df[column].isin(values)]
    
    if 'suspected' in values_cases['Status'].values:
        with st_columns[i_col]:
            status = st.selectbox(f'{entrytype} to consider', ['Only confirmed', 'Confirmed and suspected'], key=f'sus_{key}')
            i_col += 1
        filter_ = {'Only confirmed': ['confirmed'], 'Confirmed and suspected': ['confirmed', 'suspected']}[status]
        selected= values_cases[values_cases['Status'].isin(filter_)]
    else:
        selected = values_cases[values_cases['Status'] == 'confirmed']
    if len(selected):
        if cumulative == None:
            cumul_daily = st_columns[i_col].radio('',('Cumulative', 'Daily'))
            cumulative = True if cumul_daily == 'Cumulative' else False
                                    
        selected['Date'] = vdate_choice(selected['Date_confirmation'], selected['Date_entry'])
        if len(selected[index_col].dropna()):
            fig = go.Figure()  
            if colours:
                if len(colours) < len(values):
                    raise ValueError('You did not provide enough colours for your data!')
            else:
                colours = get_colours(len(values))
            for n,value in enumerate(values):
                if pd.isna(value):
                    vals = selected[selected[column].isna()]
                else:
                    vals = selected[selected[column] == value]
                data_to_plot = get_daily_count(vals, index_col)
                if cumulative:
                    data_to_plot = data_to_plot.cumsum()
                    fig.add_trace(go.Bar(x=data_to_plot.index, y=data_to_plot.values, marker_color=colours[n], name=f'{value}'))
                else:
                    fig.add_trace(go.Bar(x=data_to_plot.index, y=data_to_plot.values, marker_color=colours[n], opacity=0.75 , name=f'{value}'))
            if fig['data']:
                fig['data'][0]['showlegend'] = True
                fig.update_layout(barmode='stack')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown('Not enough data for rolling average')
        else:
            st.markdown(f'No reported {entrytype} match the search criteria.')
    else:
        st.markdown(f'No reported {entrytype} match the search criteria.')

def barstack_countries(df: pd.DataFrame, key: str = 'barstack_countries',
                       cumulative: Optional[bool] = None, **kwargs):
    """
    Note
    ----
    Plots countries selected by the user

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to start from.
    key : str
        A key for streamlit. The default is 'countries'.
    cumulative: bool, optional
        whether to plot cumulative or daily data. Default is None, and asks for user choice.
    **kwargs : 
        Any other keyword argument for the generic plot function.

    Returns
    -------
    None.

    """
    all_countries = sorted(list(set(df['Country'])))
    st_columns = st.columns(3)
    with st_columns[0]:
        sel_countries = st.multiselect('Select countries to compare', all_countries, key=f'sel_{key}')
    barstack(df, values=sel_countries, key=key, st_columns=st_columns[-2:], **kwargs)