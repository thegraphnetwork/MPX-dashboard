#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 17:59:34 2022

@author: nr
"""
from typing import Optional, Union#, Any
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from mpx_dashboard import csv_specs 
from mpx_dashboard.postprocess import pad_dates, aggr_groups
from mpx_dashboard import utils

def total_weekly_metrics(aggr: pd.DataFrame, aggr_sus: Optional[pd.DataFrame] = None,
                         do_sus: Optional[bool] = None, key='metrics') -> None:
    """
    Creates streamlit widgets for weekly metrics.
    Parameters
    ----------
    aggr : pd.DataFrame
        DataFrame with the data to plot, confirmed aggregated entrytypes
    aggr_sus : pd.DataFrame, optional
        DataFrame with the data to plot, suspected aggregated entrytypes
   index_col : str, optional
       date column to use as index. The default is csv_specs.confdatecol.
   do_sus : bool, optional
       whether to consider suspected entries. Default is None, and if suspected entries are available demands user input
   key : str, optional
       Key for streamlit, avoid doubles. The default is 'metrics'.
    Returns
    -------
    None. Displays metrics in two columns
    """
    entrytype = [c for c in aggr.columns if c.startswith('daily')][0][6:]
    if do_sus == None:
        with st.columns(4)[0]:
            status = st.selectbox(f'{entrytype} to consider', ['Only confirmed', 'Confirmed and suspected'], key=f'sus_selector_{key}')
        do_sus = True if status == 'Confirmed and suspected' else False
    try:
        daily_count = pad_dates(aggr[f'daily_{entrytype}'])
    except ValueError as e:
        if e.args[0] == 'cannot reindex on an axis with duplicate labels':
            daily_count = pad_dates(aggr_groups(aggr[f'daily_{entrytype}']))
        else:
            raise ValueError
    if aggr_sus is not None and do_sus:
        try:
            daily_count_sus = pad_dates(aggr_sus[f'daily_{entrytype}'])
        except ValueError as e:
            if e.args[0] == 'cannot reindex on an axis with duplicate labels':
                daily_count_sus = pad_dates(aggr_groups(aggr_sus[f'daily_{entrytype}']))
            else:
                raise ValueError
    last_week = daily_count.iloc[-7:].sum()
    previous_week = daily_count.iloc[-14:-7].sum()  # Risky because of delay in reporting
    tot = daily_count.sum()
    if do_sus and aggr_sus is not None:
        last_week += daily_count_sus.iloc[-7:].sum()
        previous_week += daily_count_sus.iloc[-14:-7].sum()
        tot += daily_count_sus.sum()
    cols = st.columns(2)
    cols[0].metric(label=f'Total {entrytype}', value=tot)
    cols[1].metric(label=f'{entrytype.capitalize()} in the last 7 days', value=last_week, delta=f'{last_week-previous_week}')  # convert to string because delta does not accept np.int64
            
def get_colours(n: int)  -> list:
    """
    Gets an appropriate list of n colours.
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

@st.cache(allow_output_mutation=True)
def inner_plot(aggr: pd.DataFrame, daily: bool, rolling: bool, cumulative: bool,
               values: list, plot_tot: bool,  plot_sumvals: bool, colours: list,
               entrytype: str, do_sus: bool, tot_label: str, win_type: str,
               column: str, dates_padded: bool, sumvals_label: str,
               int_: Optional[int] = None, aggr_sus: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Returns the figure to pass to streamlit. 

    Parameters
    ----------
    aggr : pd.DataFrame
        DataFrame with the data to plot, confirmed aggregated entrytypes
    dates_padded : bool
        if there are zeros for the days with no reported cases. 
    values : list
        values to consider. For instance, countries to plot as individual lines. 
    column : str
        column for 'Vvalues'. 
    cumulative : bool
        wheter to plot cumulative sum of entries.
        If True, arguments for rolling average are ignored.
    daily : bool
        whether to plot daily cases. If also rolling, uses bars instead of line.
    rolling : bool
        whether to plot rolling average. 
    plot_tot : bool
        whether to plot the values from the whole dataframe (e.g. "World" if column=="Country", both genders if column="Gender")
    tot_label : str or callable
        label for the "total" curve. 
        if callable, it calls the function with locals() as kwargs
    plot_sumvals : bool
        whether to plot the sum of the selected values f(e.g. "England + Scotland + Wales")
    sumvals_label : str or callable
        label for the "sum of selected values" curve. 
        if callable, it calls the function with locals() as kwargs
    do_sus : bool
        whether to consider suspected entries.
    win_type : str
        window type for rolling average. 
    colours : list
        list of colours. For instance plotly.express.colors.qualitative.G10.
        If None, combines plotly colours to get enough.
    entrytype : str
        the entries being plotted (cases, deaths, hospitaliastions...)
    do_sus : bool
        whther to plot suspected cases.
    int_ : int, optional
        interval for running average.
    aggr_sus : pd.DataFrame, optional
        DataFrame with the data to plot, suspected aggregated entrytypes
        
    Returns
    -------
    fig : plotly.go.Figure()
        the figure to pass to streamlit

    """
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
        data_to_plot = aggr_groups(aggr)[f'daily_{entrytype}']
        if do_sus:
            to_add = aggr_groups(aggr_sus)[f'daily_{entrytype}']
            data_to_plot = data_to_plot.add(to_add, fill_value=0)
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
                fig.add_trace(go.Scatter(x=ravg.index, y=ravg.values, marker_color=colours[colour_index], opacity=0.5, name=f'{tot_label}, rolling average'))
        colour_index += 1
    if plot_sumvals:
        data_to_plot = aggr_groups(aggr[aggr[column].isin(values)])[f'daily_{entrytype}'] if plot_tot else aggr_groups(aggr)[f'daily_{entrytype}']  # if not plot_tot we already filtered
        if do_sus:
            to_add = aggr_groups(aggr_sus[aggr_sus[column].isin(values)])[f'daily_{entrytype}'] if plot_tot else aggr_groups(aggr_sus)[f'daily_{entrytype}']  # if not plot_tot we already filtered
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
                fig.add_trace(go.Scatter(x=ravg.index, y=ravg.values, marker_color=colours[colour_index], opacity=0.5, name=f'{sumvals_label}, rolling average'))
        colour_index += 1
    for n,value in enumerate(values):
        if pd.isna(value):
            data_to_plot = aggr[aggr[column].isna()][f'daily_{entrytype}']
            if do_sus:
                to_add = aggr_sus[aggr_sus[column].isna()][f'daily_{entrytype}']
                data_to_plot = data_to_plot.add(to_add, fill_value=0)
        else:
            data_to_plot = aggr[aggr[column] == value][f'daily_{entrytype}']
            if do_sus:
                # st.dataframe(aggr_sus)
                to_add = aggr_sus[aggr_sus[column] == value][f'daily_{entrytype}']
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
                    fig.add_trace(go.Scatter(x=ravg.index, y=ravg.values, marker_color=colours[colour_index + n], opacity=0.5, name=f'{value}, rolling average'))
    return fig

def plot(aggr: pd.DataFrame, aggr_sus: Optional[pd.DataFrame] = None,
         dates_padded: bool = False,
         values: list = [], column: str =csv_specs.countrycol,
         cumulative: Optional[bool] = None,
         daily: Optional[bool] = None, rolling: Optional[bool] = None,
         plot_tot: bool = False, tot_label: Union[str, callable] = 'World',
         plot_sumvals: bool = False, sumvals_label: Union[str, callable] = 'Sum of selected countries',
         do_sus: Optional[bool] = None, key: str = 'only', min_int: int = 3,
         max_int: int = 15, default: int = 7, win_type: Optional[str] = 'exponential',
         colours: Optional[list] = None, st_columns: list = []) -> None:
    """
    Uses streamlit user input to further select the data and plots it. 
    Hence this part cannot be cached. The inner function is cached not to repeat plotting.
    
    Parameters
    ----------
    aggr : pd.DataFrame
        DataFrame with the data to plot, confirmed aggregated entrytypes
    aggr_sus : pd.DataFrame, optional
        DataFrame with the data to plot, suspected aggregated entrytypes
    dates_padded : bool
        if there are zeros for the days with no reported cases. Default is False
    values : list
        values to consider. For instance, countries to plot as individual lines. Default is [], for only total curve
    column : str
        column for 'values'. Default is csv_specs.countrycol
    cumulative : bool, optional
        wheter to plot cumulative sum of entries. Default is None. In such case, a user input is used.
        If True, arguments for rolling average are ignored.
    daily : bool, optional
        whether to plot daily cases. If also rolling, uses bars instead of line. Default is None. In such case, a user input is used.
    rolling : bool, optional
        whether to plot rolling average. Default is None. In such case, a user input is used.
    plot_tot : bool
        whether to plot the values from the whole dataframe (e.g. "World" if column=="Country", both genders if column="Gender")
    tot_label : str or callable
        label for the "total" curve. Default is World.
        if callable, it calls the function with locals() as kwargs
    plot_sumvals : bool
        whether to plot the sum of the selected values f(e.g. "England + Scotland + Wales")
    sumvals_label : str or callable
        label for the "sum of selected values" curve. Default is "sum of selected countries"
        if callable, it calls the function with locals() as kwargs
    do_sus : bool, optional
        whether to consider suspected entries. Default is None, and if suspected entries are available demands user input
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
    colours : list, optional
        list of colours. For instance plotly.express.colors.qualitative.G10.
        If None, combines plotly colours to get enough.
    st_columns : list, optional
        list of streamlit columns. If [], creates 3 columns.
        Provide empty columns, or all columns, if others are already used.
        
    Returns
    -------
    None.
    """
    if not st_columns:
        st_columns = st.columns(3)
    i_col = -(len(st_columns))
    
    entrytype = [c for c in aggr.columns if c.startswith('daily')][0][6:]
    if do_sus == None and aggr_sus is not None and len(aggr_sus):
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
        else:
            int_ = None
        fig = inner_plot(aggr, daily, rolling, cumulative, values, plot_tot,
                         plot_sumvals, colours, entrytype, do_sus, tot_label,
                         win_type, column, dates_padded, sumvals_label, int_=int_,
                         aggr_sus=aggr_sus)
        if fig['data']:
            fig['data'][0]['showlegend'] = True
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('Not enough data for rolling average')
    else:
        st.markdown(f'No reported {entrytype} match the search criteria.')

def plot_countries(aggr: pd.DataFrame, aggr_sus: Optional[pd.DataFrame] = None,
                   key: str = 'countries', cumulative: Optional[bool] = None,
                   daily: Optional[bool] = None, rolling: Optional[bool] = None,
                   do_sus: Optional[bool] = None, **kwargs) -> None:
    """
    Plots countries selected by the user

    Parameters
    ----------
    aggr : pd.DataFrame
        Dataframe to start from.
    aggr_sus : pd.DataFrame, optional
        DataFrame with the data to plot, suspected aggregated entrytypes
    key : str
        A key for streamlit. The default is 'countries'.
    cumulative : bool, optional
        wheter to plot cumulative sum of entries. Default is None. In such case, a user input is used.
        If True, arguments for rolling average are ignored.
    entrytype : str
        what the entries are (cases, deaths, ...)
    daily : bool, optional
        whether to plot daily cases. If also rolling, uses bars instead of line. Default is None. In such case, a user input is used.
    rolling : bool, optional
        whether to plot rolling average. Default is None. In such case, a user input is used.
    do_sus : bool, optional
        whether to consider suspected entries. Default is None, and if suspected entries are available demands user input
    **kwargs : 
        Any other keyword argument for the generic plot function.

    Returns
    -------
    None.
    """
    
    all_countries = sorted(list(set(aggr[csv_specs.countrycol]).union(set(aggr_sus[csv_specs.countrycol]))))
    st_columns = st.columns(4)
    with st_columns[0]:
        sel_countries = st.multiselect('Select countries to compare', all_countries + ['World', 'Sum of selected'], key=f'sel_{key}')
    plot_tot, plot_sumvals = 'World' in sel_countries, 'Sum of selected' in sel_countries
    plot(aggr, aggr_sus=aggr_sus, values=[i for i in sel_countries if i not in ['World', 'Sum of selected']],
         column=csv_specs.countrycol, key=key,plot_sumvals=plot_sumvals,
         tot_label='World', plot_tot=plot_tot, sumvals_label='Sum of selected countries',
         st_columns=st_columns[-3:], cumulative=cumulative, daily=daily, rolling=rolling,
         do_sus=do_sus, **kwargs)

def plot_tot(aggr: pd.DataFrame, aggr_sus: Optional[pd.DataFrame] = None,
             label: str = '', key: str = 'tot', colour: str = 'black',
             cumulative: Optional[bool] = None, daily: Optional[bool] = None,
             rolling: Optional[bool] = None, do_sus: Optional[bool] = None, **kwargs)  -> None:
    
    """
    plots the whole dataframe according to the selection criteria

    Parameters
    ----------
    aggr : pd.DataFrame
        DataFrame with the data to plot, confirmed aggregated entrytypes
    aggr_sus : pd.DataFrame, optional
        DataFrame with the data to plot, suspected aggregated entrytypes
    label : str
        The label for your single curve. The default is "daily"/"cumulative" entrytype.
    key : str, optional
        Key for streamlit. The default is 'tot'.
    colour : str, optional
        Colour for the curve. The default is 'black'.
    cumulative : bool, optional
        wheter to plot cumulative sum of entries. Default is None. In such case, a user input is used.
        If True, arguments for rolling average are ignored.
    daily : bool, optional
        whether to plot daily cases. If also rolling, uses bars instead of line. Default is None. In such case, a user input is used.
    rolling : bool, optional
        whether to plot rolling average. Default is None. In such case, a user input is used.
    do_sus : bool, optional
        whether to consider suspected entries. Default is None, and if suspected entries are available demands user input
    **kwargs : 
        Any other keyword argument for the generic plot function.

    Returns
    -------
    None.
    """
    
    if not label:
        label = lambda d: '{} {}'.format('cumulative' if d['cumulative'] else 'daily', d['entrytype'])
    plot(aggr, aggr_sus=aggr_sus, key=key, plot_tot=True, tot_label=label, colours=[colour],
         cumulative=cumulative, daily=daily, rolling=rolling, do_sus=do_sus, **kwargs)
    
def plot_genders(aggr: pd.DataFrame, aggr_sus: Optional[pd.DataFrame] = None,
                 key: str = 'genders', cumulative: Optional[bool] = None,
                 daily: Optional[bool] = None, rolling: Optional[bool] = None,
                 do_sus: Optional[bool] = None, **kwargs) -> None:
    """
    Plots curves for each gender

    Parameters
    ----------
    aggr : pd.DataFrame
        the dataframe to start from.
    aggr_sus : pd.DataFrame, optional
        DataFrame with the data to plot, suspected aggregated entrytypes
    key : str, optional
        A key for streamlir. The default is 'genders'.
    cumulative : bool, optional
        wheter to plot cumulative sum of entries. Default is None. In such case, a user input is used.
        If True, arguments for rolling average are ignored.
    daily : bool, optional
        whether to plot daily cases. If also rolling, uses bars instead of line. Default is None. In such case, a user input is used.
    rolling : bool, optional
        whether to plot rolling average. Default is None. In such case, a user input is used.
    do_sus : bool, optional
        whether to consider suspected entries. Default is None, and if suspected entries are available demands user input
    **kwargs : 
        Any other keyword argument for the generic plot function.

    Returns
    -------
    None.
    """
    
    plot(aggr, aggr_sus=aggr_sus, values=['male', 'female'], key=key, column=csv_specs.gendercol, plot_tot=True,
         cumulative=cumulative, daily=daily, rolling=rolling, tot_label='all cases',
         colours=['black', 'blue', 'pink'], do_sus=do_sus, **kwargs)
    
def plot_outcome(aggr: pd.DataFrame, aggr_sus: Optional[pd.DataFrame] = None,
                 include_nan: bool = True, key: str = 'outcome',
                 do_sus: Optional[bool] = None, **kwargs) -> None:
    """
    Plots the outcome of cases

    Parameters
    ----------
    aggr : pd.DataFrame
        Dataframe to start from.
    aggr_sus : pd.DataFrame, optional
        DataFrame with the data to plot, suspected aggregated entrytypes
    include_nan : bool
        Whether to plot the nans. The default is True.
    key : str
        Key for streanlit. The default is 'outcome'.
    do_sus : bool, optional
        whether to consider suspected entries. Default is None, and if suspected entries are available demands user input
    **kwargs : 
        Any other keyword argument for the generic plot function.

    Returns
    -------
    None.
    """
    
    values = csv_specs.outcomevals + [np.nan] if include_nan else csv_specs.outcomevals
    plot(aggr, aggr_sus=aggr_sus, values=values, column=csv_specs.outcomecol,
         cumulative=True, do_sus=do_sus, **kwargs)

def plot_needed_hospital(aggr: pd.DataFrame, aggr_sus: Optional[pd.DataFrame] = None,
                         include_nan: bool = True, key: str = 'hospitalised',
                         do_sus: Optional[bool] = None, **kwargs) -> None:
    """
    Plots how many cases needed hospitalisation

    Parameters
    ----------
    aggr : pd.DataFrame
        Dataframe to start from.
    aggr_sus : pd.DataFrame, optional
        DataFrame with the data to plot, suspected aggregated entrytypes
    include_nan : bool
        Whether to plot the nans. The default is True.
    key : str
        Key for streanlit. The default is 'outcome'.
    do_sus : bool, optional
        whether to consider suspected entries. Default is None, and if suspected entries are available demands user input
    **kwargs : 
        Any other keyword argument for the generic plot function.

    Returns
    -------
    None.
    """
    
    values = [True, False, np.nan] if include_nan else [True, False]
    plot(aggr, aggr_sus=aggr_sus, values=values, column=csv_specs.hospitalcol,
         cumulative=True, do_sus=do_sus, **kwargs)

def inner_barstack(aggr: pd.DataFrame, aggr_sus: pd.DataFrame, colours: Optional[list],
                   column: str, values: list, do_sus: bool, entrytype: str,
                   dates_padded: bool, cumulative: bool) -> go.Figure:
    """

    Parameters
    ----------
    aggr : pd.DataFrame
        aggregated entries per day per value.
    aggr_sus : pd.DataFrame
        aggregated suspected entries per day per value.
    colours : list, optional
        list of colours. For instance plotly.express.colors.qualitative.G10.
        If None, combines plotly colours to get enough.
    column : str
        column for 'values'
    values : list
        values to consider. For instance, countries to plot as individual lines.
    do_sus : bool
        whether to consider suspected entries.
    entrytype : str
        the type of entries provided (cases, deaths, hospitalisations...).
    dates_padded : bool
        if there are zeros for the days with no reported cases. 
    cumulative : bool
        wheter to plot cumulative sum of entries.

    Raises
    ------
    ValueError
        If fewere colours than values are provided, an error is raised.

    Returns
    -------
    fig : go.Figure
        barstacks plotly figure

    """
    fig = go.Figure()  
    if colours:
        if len(colours) < len(values):
            raise ValueError('You did not provide enough colours for your data!')
    else:
        colours = get_colours(len(values))
    for n,value in enumerate(values):
        if pd.isna(value):
            data_to_plot = aggr[aggr[column].isna()][f'daily_{entrytype}']
            if do_sus:
                to_add = aggr_sus[aggr_sus[column].isna()][f'daily_{entrytype}']
                data_to_plot = data_to_plot.add(to_add, fill_value=0)
        else:
            data_to_plot = aggr[aggr[column] == value][f'daily_{entrytype}']
            if do_sus:
                to_add = aggr_sus[aggr_sus[column] == value][f'daily_{entrytype}']
                data_to_plot = data_to_plot.add(to_add, fill_value=0)
        if not dates_padded:
            data_to_plot = pad_dates(data_to_plot)
        if cumulative:
            data_to_plot = data_to_plot.cumsum()
            fig.add_trace(go.Bar(x=data_to_plot.index, y=data_to_plot.values, marker_color=colours[n], name=f'{value}'))
        else:
            fig.add_trace(go.Bar(x=data_to_plot.index, y=data_to_plot.values, marker_color=colours[n], opacity=0.75 , name=f'{value}'))
    return fig

def barstack(aggr: pd.DataFrame, aggr_sus: Optional[pd.DataFrame] = None,
             dates_padded: bool = False, values: list = [],
             column: str = csv_specs.countrycol, cumulative: Optional[bool] = None,
             daily: bool = True, do_sus: Optional[bool] = None,
             key: str = 'barstack', colours: Optional[list] = None,
             st_columns: list = []) -> None:
    """
    Uses streamlit user input to further select the data and plots it.
    
    Parameters
    ----------
    aggr : pd.DataFrame
        DataFrame with the data to plot, confirmed aggregated entrytypes
    aggr_sus : pd.DataFrame, optional
        DataFrame with the data to plot, suspected aggregated entrytypes
    dates_padded : bool
        if there are zeros for the days with no reported cases. 
    values : list
        values to consider. For instance, countries to plot as individual lines.
    column : str
        column for 'values'. Default is csv_specs.countrycol
    cumulative : bool, Optional
        wheter to plot cumulative sum of entries. Default is None, and asks for user input.
    do_sus : bool, optional
        whether to consider suspected entries. Default is None, and if suspected entries are available demands user input
    key : str, optional
        Key for streamlit, avoid doubles. The default is 'only'.
    colours : list, optional
        list of colours. For instance plotly.express.colors.qualitative.G10.
        If None, combines plotly colours to get enough.
    st_columns : list, optional
        list of streamlit columns. If [], creates 2 columns.
        Provide empty columns, or all columns, if others are already used.
    
    Returns
    -------
    None.
    """
    
    if not st_columns:
        st_columns = st.columns(2)
    i_col = -(len(st_columns))
    
    entrytype = [c for c in aggr.columns if c.startswith('daily')][0][6:]
    
    if do_sus == None and aggr_sus is not None and len(aggr_sus):
        with st_columns[i_col]:
            status = st.selectbox(f'{entrytype} to consider', ['Only confirmed', 'Confirmed and suspected'], key=f'sus_{key}')
            i_col += 1
        do_sus = True if status == 'Confirmed and suspected' else False
        
    aggr.index.name = 'Date'
    aggr = aggr[aggr[column].isin(values)]
    if do_sus:
        aggr_sus.index.name = 'Date'
        aggr_sus =  aggr_sus[aggr_sus[column].isin(values)]
    if len(aggr) or do_sus and len(aggr_sus):
        if cumulative == None:
            cumul_daily = st_columns[i_col].radio('',('Cumulative', 'Daily'))
            cumulative = True if cumul_daily == 'Cumulative' else False
        fig = inner_barstack(aggr, aggr_sus, colours, column, values, do_sus,
                             entrytype, dates_padded, cumulative)
        if fig['data']:
            fig['data'][0]['showlegend'] = True
            fig.update_layout(barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('Not enough data for rolling average')
    else:
        st.markdown(f'No reported {entrytype} match the search criteria.')

def barstack_countries(aggr: pd.DataFrame, aggr_sus: Optional[pd.DataFrame] = None,
                       key: str = 'barstack_countries', cumulative: Optional[bool] = None,
                       do_sus: Optional[bool] = None, **kwargs) -> None:
    """
    Plots countries selected by the user

    Parameters
    ----------
    aggr : pd.DataFrame
        DataFrame with the data to plot, confirmed aggregated entrytypes
    aggr_sus : pd.DataFrame, optional
        DataFrame with the data to plot, suspected aggregated entrytypes
    key : str
        A key for streamlit. The default is 'countries'.
    cumulative : bool, optional
        whether to plot cumulative or daily data. Default is None, and asks for user choice.
    do_sus : bool, optional
        whether to consider suspected entries. Default is None, and if suspected entries are available demands user input
    **kwargs : 
        Any other keyword argument for the generic plot function.

    Returns
    -------
    None.
    """
    
    all_countries = sorted(list(set(aggr[csv_specs.countrycol]).union(set(aggr_sus[csv_specs.countrycol]))))
    st_columns = st.columns(3)
    with st_columns[0]:
        sel_countries = st.multiselect('Select countries to compare', all_countries, key=f'sel_{key}')
    barstack(aggr, aggr_sus=aggr_sus, values=sel_countries, key=key, st_columns=st_columns[-2:],do_sus=do_sus, **kwargs)

@st.cache(allow_output_mutation=True)
def country_to_iso3_with(aggr: pd.DataFrame) -> pd.DataFrame:
    """
    Obtains ISO3 from countries and groups "countries" within the same ISO3 (.e.g. England, Wales, Scotland, Northern Ireland)
    Parameters
    ----------
    aggr : pd.DataFrame
        aggregated data (entries per day per country).

    Returns
    -------
    pd.DataFrame
        properly aggregated data with ISO3

    """
    aggr[csv_specs.iso3col] = aggr[csv_specs.countrycol].map(utils.country_to_iso3)
    return aggr.groupby(csv_specs.iso3col).resample('D').sum(numeric_only=True)

@st.cache(allow_output_mutation=True)
def cached_choropleth(aggr: pd.DataFrame, locations: str, color: str, hover_name: str,
                    color_continuous_scale: list, animation_frame: str,
                    range_color: Optional[list]) -> go.Figure:
    """
    Simply a cached version of px.choropleth

    Parameters
    ----------
    aggr : pd.DataFrame
        entries per day per country
    locations : str
        column with the ISO3
    color : str
        column to use for the coloring (column with daily/cumulative entries)
    hover_name : str
        country name to display if hovering.
    color_continuous_scale : list
        color scale
    animation_frame : str
        the date to animate the choropleth
    range_color : Optional[list]
        the color range (if fixed)

    Returns
    -------
    go.Figure
        animated map

    """
    return px.choropleth(aggr, locations=locations, color=color, hover_name=hover_name,
                        color_continuous_scale=color_continuous_scale,
                        animation_frame=animation_frame, range_color=range_color)

def evolution_on_map(aggr: pd.DataFrame,  aggr_sus: Optional[pd.DataFrame] = None,
                     location_col: str = csv_specs.countrycol,
                     curve: Optional[str] = None, dateslice: Optional[tuple] = None,
                     do_sus: Optional[bool] = None, key: str = 'map',
                     st_columns: Optional[list] = None, min_int: int = 3, max_int: int = 15,
                     default: int = 7, win_type: Optional[str] = 'exponential',
                     color_scale: Union[str, list] = None) -> None:
    """
    Plots entries on  a map
    Parameters
    ----------
    aggr : pd.DataFrame
        DataFrame with the data to plot, confirmed aggregated entrytypes
    aggr_sus : pd.DataFrame, optional
        DataFrame with the data to plot, suspected aggregated entrytypes
    location_col : str, optional
        column for location data. Default uses country and turns it into an ISO3
    curve : str, optional
        cumulative, daily, or rolling average. If None produces radio selector
    dateslice : tuple, optional
        minimum and maximum date to consider. Default(None) creates date inputs
    do_sus : bool, optional
        whether to consider suspected entries. Default is None, and if suspected entries are available demands user input
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
    
    if not st_columns:
        st_columns = st.columns(4)
    i_col = -(len(st_columns))
    entrytype = [c for c in aggr.columns if c.startswith('daily')][0][6:]
    if do_sus == None and aggr_sus is not None and len(aggr_sus):
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
    if location_col == csv_specs.countrycol:
        aggr = country_to_iso3_with(aggr)
        if do_sus:
            aggr_sus = country_to_iso3_with(aggr_sus)
    all_isos = sorted(list(set(aggr.index.get_level_values(csv_specs.iso3col))))
    if do_sus:
        all_isos = all_isos.union(set(aggr_sus.index.get_level_values(csv_specs.iso3col)))
    new_index = pd.MultiIndex.from_product([all_isos, date_range], names=[csv_specs.iso3col, 'Date'])
    aggr = aggr.reindex(new_index)
    aggr[f'daily {entrytype}'] = aggr[f'daily {entrytype}'].fillna(0)
    if do_sus:
        aggr_sus.index.name = 'Date'
        aggr_sus = aggr_sus.rename(columns={f'daily_{entrytype}': f'daily {entrytype}'})
        aggr_sus = aggr_sus.reindex(new_index)
        aggr_sus[f'daily {entrytype}'] = aggr_sus[f'daily {entrytype}'].fillna(0)
        aggr += aggr_sus
    if curve == 'cumulative':
        aggr[f'cumulative {entrytype}'] = aggr.groupby(csv_specs.iso3col).cumsum()[f'daily {entrytype}']
    aggr = aggr.reset_index()
    aggr[csv_specs.countrycol] = aggr[csv_specs.iso3col].map(utils.iso3_to_country)
    aggr['Date'] = aggr['Date'].map(lambda x: x.strftime('%d-%m-%Y'))
    curve_col = f'{curve} {entrytype if curve != "rolling average" else ""}'.strip()
    if curve_col == 'rolling average':
        aggr['rolling average'] =  aggr[f'daily {entrytype}'].rolling(int_, win_type=win_type).mean()
    fig = cached_choropleth(aggr, locations=csv_specs.iso3col,
                        color=curve_col, 
                        hover_name=csv_specs.countrycol,
                        color_continuous_scale=color_scale if color_scale else px.colors.sequential.Plasma,
                        animation_frame='Date',
                        range_color=[0, aggr[curve_col].max()] if fix_scale else None)
    st.plotly_chart(fig, use_container_width=True)

@st.cache(allow_output_mutation=True)
def delay_distr_pre(df: pd.DataFrame, date_col1: str = csv_specs.confdatecol,
                date_col2: str = csv_specs.entrydatecol) -> pd.Series:
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
    pd.Series
        the difference (not NA values) between date_col1 and date_col2 in days
    """
    notna = df[np.logical_and(df[date_col1].notna(), df[date_col2].notna())]
    series = notna[date_col2] - notna[date_col1]
    return series.apply(lambda x: x.days)

@st.cache(allow_output_mutation=True)
def delay_distr_inner(series: pd.Series, min_: int, max_: int) -> go.Figure:
    """

    Parameters
    ----------
    series : pd.Series
        difference between two date columns (cf. delay_distr_pre)
    min_ : int
        minimum value to consider.
    max_ : int
        maximum value to consider.

    Returns
    -------
    go.Figure
        the histogram of series.

    """
    series = series[np.logical_and(series > min_ , series < max_)]
    return px.histogram(series)

def delay_distr(df: pd.DataFrame, date_col1: str = csv_specs.confdatecol,
                date_col2: str = csv_specs.entrydatecol) -> None:
    """
    Plots distribution of date difference between two date columns.
    NB. Works on disaggregated/unaggregated data!
    
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
    series = delay_distr_pre(df, date_col1, date_col2)
    st.markdown('Exclude extreme values')
    left, right = st.columns(2)
    min_ = left.number_input("Minimum", min_value=series.min(), max_value=series.max(), value=-5, step=1)
    max_ = right.number_input("Maximum", min_value=series.min(), max_value=series.max(), value=10, step=1)
    st.plotly_chart(delay_distr_inner(series, min_, max_), use_container_width=True)
