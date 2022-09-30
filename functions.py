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
from typing import Optional, Union#, Any
import csv_specs
import pyarrow.parquet as pq
import pyarrow as pa
import utils

# date_choice = lambda x, y: y if pd.isna(x) else y
# vdate_choice = np.vectorize(date_choice)

# @st.cache(allow_output_mutation=True)
# def old_load_data(file: str, nrows: Optional[int] = None, cols: Optional[list] = None,
#               maptobool: Optional[list] = None, errors: str = 'raise', **kwargs):
#     """
#     Parameters
#     ----------
#     file: str
#         the csv to read
#     nrows : Optional[int], optional
#         number of rows to load. The default is None, i.e. all rows.
#     cols : Optional[list], optional
#         columns to read. The default is None, i.e. all rows.
#     maptobool: Optional[list]
#         columns to map to a "boolean" (True, False, NaN)
#     errors: str
#         what to do with errors (raise, coerce, ignore), as in pd.to_datetime. Default is 'coerce'.
#     kwargs: dict
#         additional kwargs to pass to read_csv. e.g. dtype
#     Returns
#     -------
#     data : pd.DataFrame
#         the DataFrame of desired rows and columns, dates are turned to pd.datetime.
#     """
    
#     data = pd.read_csv(file, nrows=nrows, usecols=cols, **kwargs)
#     for c in maptobool:
#         data[c] = data[c].map({'Y': True, 'N': False, 'NA': None})
#     for c in data.columns:
#         if c.startswith('Date_'):
#             data[c] = pd.to_datetime(data[c], errors=errors) 
#     return data 

# @st.cache(allow_output_mutation=True)
def load_cases(file: str, skiprows: Optional[int] = None, nrows: Optional[int] = None,
               usecols: Optional[list] = None, maptobool: Optional[list] = None, **kwargs):
    """
    Parameters
    ----------
    file: str
        the csv to read
    skiprows: int, optional
        the number of rows to skip
    nrows : int, optional
        number of rows to load. The default is None, i.e. all rows.
    usecols : list, optional
        columns to read. The default is None, i.e. all rows.
    maptobool: list, optional
        columns to map to a "pseudoboolean" (True, False, NaN)
    kwargs: dict
        additional kwargs to pass to read_csv. e.g. dtype
    Returns
    -------
    cases : pd.DataFrame
        the DataFrame of desired rows and columns, dates are turned to pd.datetime.
    """
    if 'parse_date' not in kwargs:
        if usecols == None:
            usecols = pd.read_csv(file, nrows=0).columns
        kwargs['parse_dates'] =  csv_specs.datecols if csv_specs.datecols else [c for c in usecols if csv_specs.isdatecol(c)]
    if 'dtype' not in kwargs:
        if usecols is None:
            usecols = pd.read_csv(file, nrows=0).columns
        kwargs['dtype'] = {c: csv_specs.guess_dtype(c) for c in usecols}
    if type(skiprows) == int:
        skiprows = range(1, skiprows+1)
    cases = pd.read_csv(file, skiprows=skiprows, nrows=nrows, usecols=usecols,
                        **kwargs)
    if maptobool is None:
        maptobool = [c for c in usecols if csv_specs.ispseudobool(c)]
    for c in maptobool:
            cases[c] = cases[c].map(csv_specs.pseudobooldict)
    return cases 

def group_and_aggr(df: pd.DataFrame, column: str = csv_specs.countrycol, date_col: str = csv_specs.confdatecol,
                   dropna: bool = True, entrytype: str = 'cases',dropzeros: bool = True):
    """
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with all the cases (one per row).
    column : str, optional
        The level to group by. The default is csv_specs.countrycol.
    date_col : str, optional
        The date to use as index, cases are summed for each day based on this column. The default is csv_specs.confdatecol.
    dropna : bool, optional
        Whether to keep Nans. The default is True.
    entrytype : str, optional
        The type of entry. The default is 'cases'.
    dropzeros : bool, optional
        Whether to drop days on which there are no reported entrytype. The default is True.

    Returns
    -------
    agg : TYPE
        DESCRIPTION.

    """
    daily_col = f'daily_{entrytype}'
    df = df[df[date_col].notna()].set_index(date_col)
    count_col = df.columns[0]
    agg = (df.groupby(column, observed=True, dropna=dropna)[count_col]
           .resample('D')
           .count()
           .rename(daily_col)
           .reset_index(level=0)
           )
    if dropzeros:
            agg = agg.loc[agg[daily_col] != 0]
    return agg

def add_to_parquet(df: pd.DataFrame, path: str):
    """
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save as parquet
    path : str
        the path for the file.

    Returns
    -------
    None.

    """
    pq.write_to_dataset(pa.Table.from_pandas(df) , root_path=path)

aggr_groups = lambda df: df.groupby(level=0).sum().asfreq('D').fillna(0)  # e.g. aggregates all countries for each day

def pad_dates(df: Union[pd.Series, pd.DataFrame], **kwargs):
    """
    Parameters
    ----------
    df : pd.Series, pd.DataFrame
        series of dataframe whose index values are dates
    **kwargs : dict
        any argument to pass to fillna (generally value)
        if no 'value' is provided, value=0 is used.

    Returns
    -------
    pd.Series or pd.DataFrame
        the original data but index values with the missing dates are introduced and filled with value(default is 0)

    """
    if not kwargs:
        kwargs = dict(value=0)
    return df.asfreq('D').fillna(**kwargs)

@st.cache()
def get_daily_count(df: pd.DataFrame, index_col: str):
    """
    Parameters
    ----------
    df : pd.DataFrame
        dataframe with a case per row
    index_col : str
        date column to use as index

    Returns
    -------
    pd.Series
        The count of entries for each day.
    """
    count_col = df.columns[0]
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

def total_weekly_metrics(aggr: pd.DataFrame, entrytype: str = 'cases',
                         index_col: str = csv_specs.confdatecol):
    """
    Parameters
    ----------
    df : pd.DataFrame
        dataframe with one entrytype per line
    entrytype : str, optional
       cases/deaths/hospitalisations. The default is 'cases'.
   index_col : str, optional
       date column to use as index. The default is csv_specs.confdatecol.

    Returns
    -------
    None. Displays metrics in two columns
    """
    
    daily_count = pad_dates(aggr[f'daily_{entrytype}'])
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

def plot(aggr: pd.DataFrame, aggr_sus: Optional[pd.DataFrame] = None,
         dates_padded: bool = False,
         values: list = [], column: str =csv_specs.countrycol,
         cumulative: Optional[bool] = None,
         daily: Optional[bool] = None, rolling: Optional[bool] = None,
         plot_tot: bool = False, tot_label: Union[str, callable] = 'World',
         plot_sumvals: bool = False, sumvals_label: Union[str, callable] = 'Sum of selected countries',
         key: str = 'only', min_int: int = 3,
         max_int: int = 15, default: int = 7, win_type: Optional[str] = 'exponential',
         colours: Optional[list] = None, st_columns: list = []):
    """
    Note
    ----
    Uses streamlit user input to further select the data and plots it.
    
    Parameters
    ----------
    aggr : pd.DataFrame
        DataFrame with the data to plot, confirmed aggregated entrytypes
    aggr_sus: pd.DataFrame, optional
        DataFrame with the data to plot, suspected aggregated entrytypes
    dates_padded: bool
        if there are zeros for the days with no reported cases. Default is False
    values : list
        values to consider. For instance, countries to plot as individual lines. Default is [], for only total curve
    column: str
        column for 'values'. Default is csv_specs.countrycol
    cumulative : bool, optional
        wheter to plot cumulative sum of entries. Default is None. In such case, a user input is used.
        If True, arguments for rolling average are ignored.
    daily: bool, optional
        whether to plot daily cases. If also rolling, uses bars instead of line. Default is None. In such case, a user input is used.
    rolling: bool, optional
        whether to plot rolling average. Default is None. In such case, a user input is used.
    plot_tot: bool
        whether to plot the values from the whole dataframe (e.g. "World" if column=="Country", both genders if column="Gender")
    tot_label: str or callable
        label for the "total" curve. Default is World.
        if callable, it calls the function with locals() as kwargs
    plot_sumvals: bool
        whether to plot the sum of the selected values f(e.g. "England + Scotland + Wales")
    sumvals_label: str or callable
        label for the "sum of selected values" curve. Default is "sum of selected countries"
        if callable, it calls the function with locals() as kwargs
    key : str, optional
        Key for streamlit, avoid doubles. The default is 'only'.
    min_int : int, optional
        minimum value for rolling average. The default is 3.
    max_int : int, optional
        maximum value for rolling average. The default is 15.
    default : int, optional
        default value for rolling average number input. The default is 7.
    win_type : Optional[str], optional
        window type for rolling average. The default is exponential.
    colours: list, optional
        list of colours. For instance plotly.express.colors.qualitative.G10.
        If None, combines plotly colours to get enough.
    st_columns: list, optional
        list of streamlit columns. If [], creates 3 columns.
        Provide empty columns, or all columns, if others are already used.
        
    Returns
    -------
    None.
    """
    if not st_columns:
        st_columns = st.columns(3)
    i_col = -3
    
    entrytype = [c for c in aggr.columns if c.startswith('daily')][0][6:]
    if aggr_sus is not None and len(aggr_sus):
        with st_columns[i_col]:
            status = st.selectbox(f'{entrytype} to consider', ['Only confirmed', 'Confirmed and suspected'], key=f'sus_{key}')
        i_col += 1
        do_sus = True if status == 'Confirmed and suspected' else False
    
    aggr.index.name = 'Date'
    aggr = aggr if plot_tot else aggr[aggr[column].isin(values)]
    if do_sus:
        aggr_sus.index.name = 'Date'
        aggr_sus =  aggr_sus if plot_tot else aggr_sus[aggr_sus[column].isin(values)]
    if len(aggr) or do_sus and len(aggr_sus):
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
                data_to_plot = aggr_groups(aggr)
                if do_sus:
                    to_add = aggr_groups(aggr_sus)
                    data_to_plot = data_to_plot.add(to_add, fill_value=0)
                if not dates_padded:
                    data_to_plot = pad_dates(data_to_plot)
                if callable(tot_label):
                    tot_label = tot_label(locals())
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
                data_to_plot = aggr_groups(aggr[aggr[column].isin(values)]) if plot_tot else aggr_groups(aggr)  # if not plot_tot we already filtered
                if do_sus:
                    to_add = aggr_groups(aggr_sus[aggr_sus[column].isin(values)]) if plot_tot else aggr_groups(aggr_sus)  # if not plot_tot we already filtered
                    data_to_plot = data_to_plot.add(to_add, fill_value=0)
                if not dates_padded:
                    data_to_plot = pad_dates(data_to_plot)
                if callable(sumvals_label):
                    sumvals_label = sumvals_label(locals())
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
                    data_to_plot = aggr[aggr[column].isna()]
                    if do_sus:
                        to_add = aggr_sus[aggr_sus[column].isna()]
                        data_to_plot = data_to_plot.add(to_add, fill_value=0)
                else:
                    data_to_plot = aggr[aggr[column] == value]
                    if do_sus:
                        to_add = aggr_sus[aggr_sus[column == value]]
                        data_to_plot = data_to_plot.add(to_add, fill_value=0)
                if not dates_padded:
                    data_to_plot = pad_dates(data_to_plot)
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
    
def plot_countries(aggr: pd.DataFrame, key: str = 'countries',
                   cumulative: Optional[bool] = None, daily: Optional[bool] = None,
                   rolling: Optional[bool] = None, **kwargs):
    """
    Note
    ----
    Plots countries selected by the user

    Parameters
    ----------
    aggr : pd.DataFrame
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
    
    all_countries = sorted(list(set(aggr[csv_specs.countrycol])))
    st_columns = st.columns(4)
    with st_columns[0]:
        sel_countries = st.multiselect('Select countries to compare', all_countries + ['World', 'Sum of selected'], key=f'sel_{key}')
    plot_tot, plot_sumvals = 'World' in sel_countries, 'Sum of selected' in sel_countries
    plot(aggr, values=[i for i in sel_countries if i not in ['World', 'Sum of selected']],
         column=csv_specs.countrycol, key=key,plot_sumvals=plot_sumvals,
         tot_label='World', plot_tot=plot_tot, sumvals_label='Sum of selected countries',
         st_columns=st_columns[-3:], cumulative=cumulative, daily=daily, rolling=rolling,
         **kwargs)

def plot_tot(aggr: pd.DataFrame, label: str = '', entrytype: str = 'cases', key: str = 'tot',
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
        label = lambda d: '{} {}'.format('cumulative' if d['cumulative'] else 'daily', d['entrytype'])
    plot(aggr, key=key, plot_tot=True, tot_label=label, colours=[colour],
         entrytype=entrytype, cumulative=cumulative, daily=daily, rolling=rolling, **kwargs)
    
def plot_genders(aggr: pd.DataFrame, key: str = 'genders', cumulative: Optional[bool] = None,
                 daily: Optional[bool] = None, rolling: Optional[bool] = None, **kwargs):
    """
    Note
    ----
    Plots curves for each gender

    Parameters
    ----------
    aggr : pd.DataFrame
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
    
    plot(aggr, values=['male', 'female'], key=key, column=csv_specs.gendercol, plot_tot=True,
         cumulative=cumulative, daily=daily, rolling=rolling, tot_label='all cases',
         colours=['black', 'blue', 'pink'], **kwargs)
    
def plot_outcome(aggr: pd.DataFrame, include_nan: bool = True, key: str = 'outcome', **kwargs):
    """
    Note
    ----
    Plots the outcome of cases

    Parameters
    ----------
    aggr : pd.DataFrame
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
    
    values = csv_specs.outcomevals + [np.nan] if include_nan else csv_specs.outcomevals
    plot(aggr, values=values, column=csv_specs.outcomecol, cumulative=True, **kwargs)

def plot_needed_hospital(aggr: pd.DataFrame, include_nan: bool = True, key: str = 'hospitalised', **kwargs):
    """
    Note
    ----
    Plots how many cases needed hospitalisation

    Parameters
    ----------
    aggr : pd.DataFrame
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
    
    values = [True, False, np.nan] if include_nan else [True, False]
    plot(aggr, values=values, column=csv_specs.hospitalcol, cumulative=True, **kwargs)

def barstack(aggr: pd.DataFrame, aggr_sus: Optional[pd.DataFrame] = None, dates_padded: bool = False, values: list = [], column: str =csv_specs.countrycol,
         cumulative: Optional[bool] = None,
         daily: bool = True, key: str = 'barstack',
         colours: Optional[list] = None, st_columns: list = []):
    """
    Note
    ----
    Uses streamlit user input to further select the data and plots it.
    
    Parameters
    ----------
    aggr : pd.DataFrame
        DataFrame with the data to plot, confirmed aggregated entrytypes
    aggr_sus: pd.DataFrame, optional
        DataFrame with the data to plot, suspected aggregated entrytypes
    values : list
        values to consider. For instance, countries to plot as individual lines. Default is [], for only total curve
    column: str
        column for 'values'. Default is csv_specs.countrycol
    cumulative : bool, Optional
        wheter to plot cumulative sum of entries. Default is None, and asks for user input.
        If True, arguments for rolling average are ignored.
    entrytype: str
        what the entries are (cases, deaths, ...)
    key : str, optional
        Key for streamlit, avoid doubles. The default is 'only'.
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
    
    entrytype = [c for c in aggr.columns if c.startswith('daily')][0][6:]
    
    if aggr_sus is not None and len(aggr_sus):
        with st_columns[i_col]:
            status = st.selectbox(f'{entrytype} to consider', ['Only confirmed', 'Confirmed and suspected'], key=f'sus_{key}')
            i_col += 1
        do_sus = True if status == 'Confirmed and suspected' else False
        
    aggr.index.name = 'Date'
    aggr = aggr if plot_tot else aggr[aggr[column].isin(values)]
    if do_sus:
        aggr_sus.index.name = 'Date'
        aggr_sus =  aggr_sus if plot_tot else aggr_sus[aggr_sus[column].isin(values)]
    if len(aggr) or do_sus and len(aggr_sus):
        if cumulative == None:
            cumul_daily = st_columns[i_col].radio('',('Cumulative', 'Daily'))
            cumulative = True if cumul_daily == 'Cumulative' else False
        fig = go.Figure()  
        if colours:
            if len(colours) < len(values):
                raise ValueError('You did not provide enough colours for your data!')
        else:
            colours = get_colours(len(values))
        for n,value in enumerate(values):
            if pd.isna(value):
                data_to_plot = aggr[aggr[column].isna()]
                if do_sus:
                    to_add = aggr_sus[aggr_sus[column].isna()]
                    data_to_plot = data_to_plot.add(to_add, fill_value=0)
            else:
                data_to_plot = aggr[aggr[column] == value]
                if do_sus:
                    to_add = aggr_sus[aggr_sus[column == value]]
                    data_to_plot = data_to_plot.add(to_add, fill_value=0)
            if not dates_padded:
                data_to_plot = pad_dates(data_to_plot)
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

def barstack_countries(aggr: pd.DataFrame, key: str = 'barstack_countries',
                       cumulative: Optional[bool] = None, **kwargs):
    """
    Note
    ----
    Plots countries selected by the user

    Parameters
    ----------
    aggr : pd.DataFrame
        DataFrame with the data to plot, confirmed aggregated entrytypes
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
    
    all_countries = sorted(list(set(aggr[csv_specs.countrycol])))
    st_columns = st.columns(3)
    with st_columns[0]:
        sel_countries = st.multiselect('Select countries to compare', all_countries, key=f'sel_{key}')
    barstack(aggr, values=sel_countries, key=key, st_columns=st_columns[-2:], **kwargs)
    
def evolution_on_map(aggr: pd.DataFrame,  aggr_sus: Optional[pd.DataFrame] = None, 
                     curve: Optional[str] = None, dateslice: Optional[tuple] = None, key: str = 'map',
                     st_columns: Optional[list] = None, min_int: int = 3, max_int: int = 15,
                     default: int = 7, win_type: Optional[str] = 'exponential', color_scale: Union[str, list] = None):
    """
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data to plot, aggregated by country and date
    curve: str, optional
        cumulative, daily, or rolling average. If None produces radio selector
    dateslice: tuple, optional
        minimum and maximum date to consider. Default(None) creates date inputs
    key : str, optional
        Key for streamlit, avoid doubles. The default is 'only'.
        column to use as x axes. Default is date_confirmation if available and date_entry otherwise
    min_int : int, optional
        minimum value for rolling average. The default is 3.
    max_int : int, optional
        maximum value for rolling average. The default is 15.
    default : int, optional
        default value for rolling average number input. The default is 7.
    win_type : Optional[str], optional
        window type for rolling average. The default is exponential.
    color_scale : str or list, optional
        A color scale. The default is None, which uses 'Plasma'.
   
    Returns
    -------
    None.
    """
    
    all_countries = sorted(list(set(aggr[csv_specs.countrycol])))
    if not st_columns:
        st_columns = st.columns(4)
    i_col = -4
    entrytype = [c for c in aggr.columns if c.startswith('daily')][0][6:]
    if aggr_sus is not None and len(aggr_sus):
        with st_columns[i_col]:
            status = st.selectbox(f'{entrytype} to consider', ['Only confirmed', 'Confirmed and suspected'], key=f'sus_{key}')
            i_col += 1
        do_sus = True if status == 'Confirmed and suspected' else False
    aggr.index.name = 'Date'
    if curve == None:
        curve = st_columns[i_col].radio(f'{entrytype} to display cases', ('Cumulative', 'Daily', 'Rolling average'), key=f'radio_{key}').lower()
        i_col += 1
    if curve == 'rolling average':
        int_ = st_columns[i_col].number_input('Rolling average interval', min_value=min_int, max_value=max_int, value=default, key=key)
    with st_columns[i_col]:
        fix_scale = st.checkbox('Fix color scale', value=True, key='fix_scale')
        i_col += 1
      
    aggr = aggr.rename(columns={f'daily_{entrytype}': f'daily {entrytype}'})
    dmin, dmax = aggr.index.min(), aggr.index.max()
    if do_sus:
        dmin, dmax = min(dmin, aggr_sus.index.min()), max(dmax, aggr_sus.index.max())
    if dateslice == None:
        with st_columns[i_col]:
            dmin = st.date_input(label='Start: ', value=dmin,  # two separate widgets otherwise error while 1 date only is chosen
                            key='start', help="The start date")
            dmax = st.date_input(label='End : ', value=dmax,
                            key='end', help="The end date")
    date_range = pd.date_range(start=dmin, end=dmax, freq='D')
    new_index = pd.MultiIndex.from_product([all_countries, date_range], names=[csv_specs.countrycol, 'Date'])
    aggr = aggr.set_index(csv_specs.countrycol, append=True).reindex(new_index)
    aggr[f'daily {entrytype}'] = aggr[f'daily {entrytype}'].fillna(0)
    if do_sus:
        aggr_sus.index.name = 'Date'
        aggr_sus = aggr_sus.rename(columns={f'daily_{entrytype}': f'daily {entrytype}'})
        aggr_sus = aggr_sus.set_index(csv_specs.countrycol, append=True).reindex(new_index)
        aggr_sus[f'daily {entrytype}'] = aggr_sus[f'daily {entrytype}'].fillna(0)
        aggr += aggr_sus
    aggr['Country_ISO3'] = aggr.index.get_level_values(csv_specs.countrycol).map(utils.d_iso3)
    if curve == 'cumulative':
        aggr[f'cumulative {entrytype}'] = aggr.groupby(csv_specs.countrycol).cumsum()[f'daily {entrytype}']
    aggr = aggr.reset_index()
    aggr['Date'] = aggr['Date'].map(lambda x: x.strftime('%d-%m-%Y'))
    curve_col = f'{curve} {entrytype if curve != "rolling average" else ""}'.strip()
    if curve_col == 'rolling average':
        aggr['rolling average'] =  aggr[f'daily {entrytype}'].rolling(int_, win_type=win_type).mean()
        
    fig = px.choropleth(aggr, locations='Country_ISO3',
                        color=curve_col, 
                        hover_name=csv_specs.countrycol,
                        color_continuous_scale=color_scale if color_scale else px.colors.sequential.Plasma,
                        animation_frame='Date',
                        range_color=[0, aggr[curve_col].max()] if fix_scale else None)
    st.plotly_chart(fig, use_container_width=True)

def delay_distr(df: pd.DataFrame, date_col1: str = csv_specs.confdatecol,
                date_col2: str = csv_specs.entrydatecol):
    """
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe with all the cases.
    date_col1 : str
        the column containing the first date. The default is csv_specs.confdatecol.
    date_col2 : str
        the column containing the second date. The default is 'Date_entry'.

    Returns
    -------
    None.

    """
    notna = df[np.logical_and(df[date_col1].notna(), df[date_col2].notna())]
    series = notna[date_col2] - notna[date_col1]
    series = series.apply(lambda x: x.days)
    st.markdown('Exclude extreme values')
    left, right = st.columns(2)
    min_ = left.number_input("Minimum", min_value=series.min(), max_value=series.max(), value=-5, step=1)
    max_ = right.number_input("Maximum", min_value=series.min(), max_value=series.max(), value=10, step=1)
    series = series[np.logical_and(series > min_ ,series < max_)]

    fig = px.histogram(series)
    st.plotly_chart(fig, use_container_width=True)
