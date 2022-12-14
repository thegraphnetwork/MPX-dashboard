#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 17:58:58 2022

@author: nr
"""
import streamlit as st
import pandas as pd
from typing import Optional, Union, Any
from mpx_dashboard import csv_specs 
from epigraphhub.settings import env
from epigraphhub.connection import get_engine

def group_and_aggr(df: pd.DataFrame, column: str = csv_specs.countrycol, date_col: str = csv_specs.confdatecol,
                   dropna: bool = True, entrytype: str = 'cases',dropzeros: bool = True) -> pd.DataFrame:
    """
    Aggregates data based on a specific column (generally country) and a date column (reporting, confirmation, death...)
    
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
    agg : pd.DataFrame
        the aggregated dataframe (reported entries per date per country).

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

def aggr_groups(df: pd.DataFrame, dates_padded: bool = True, fill_value: Any = 0) -> pd.DataFrame:
    """
    Aggregates all groups (but not dates) in a dataframe. e.g. global cases by aggregating countries
    
    Parameters
    ----------
    df : pd.DataFrame
        the dataframe aggregated by groups
    dates_padded : bool, optional
        Whether the dates should be padded or not in the return value. The default is True.
    fill_value : Any, optional
        value for fillna. The default is 0. Use None to keep NaN.
        Only matters if dates_padded == True

    Returns
    -------
    df : pd.DataFrame
        the dataframe where all the groups have been aggregated together

    """
    df = df.groupby(level=0).sum(numeric_only=True)
    if dates_padded:
        df = df.asfreq('D')
        if fill_value != None:
            df = df.fillna(fill_value)  
    return df
    
def pad_dates(df: Union[pd.Series, pd.DataFrame], **kwargs) -> Union[pd.DataFrame, pd.Series]:
    """
    avoids gaps in date ranges by resampling daily and putting a desired value as filling NAs
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

###############################################################################
### Utilities, no longer used
###############################################################################

def get_locality_names() -> pd.DataFrame:
    """
    Returns
    -------
    pd.DataFrame
        description of the location_key values (google API)

    """
    return pd.read_sql_table('locality_names_0',
                             get_engine(env.db.default_credential),
                             schema = 'google_health')

def location_name_to_location_key(loc: str, loc_table: Optional[pd.DataFrame] = None) -> str:
    """
    Tries to convert location name to location key, starting from countries and going to smaller entities.
    N. B. it may have issues due to different spelling or language (Cabo Verde/Cape Verde) and multiple places with same name
    (cities named as regions, etc...). Consider as helpful but not infallible.
    
    Parameters
    ----------
    loc : str
        location name.

    Returns
    -------
    str
        location id (as in google API).

    """
    if loc_table is None:
        loc_table = get_locality_names()
    locs0 = loc_table[loc_table['aggregation_level'] == 0]
    if loc in locs0['country_name'].values:
        return locs0[locs0['country_name'] == loc]['location_key'].iloc[0]
    locs1 = loc_table[loc_table['aggregation_level'] == 1]
    if loc in locs1['subregion1_name'].values:
        return locs1[locs1['subregion1_name'] == loc]['location_key'].iloc[0]
    locs2 = loc_table[loc_table['aggregation_level'] == 2]
    if loc in locs2['subregion2_name'].values:
        return locs2[locs2['subregion2_name'] == loc]['location_key'].iloc[0]
    locs3 = loc_table[loc_table['aggregation_level'] == 3]
    if loc in locs3['locality_name'].values:
        return locs3[locs3['locality_name'] == loc]['location_key'].iloc[0]

def get_iso2_to_iso3() -> dict:
    """
    Legacy, now this dictionary is available in utils

    Returns
    -------
    dict
        dictionary of iso2 to iso3.

    """
    df = get_locality_names()
    return cols_to_dict(df, 'iso_3166_1_alpha_2', 'iso_3166_1_alpha_3')

@st.cache()
def get_daily_count(df: pd.DataFrame, index_col: str)  -> pd.Series:
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

def na_percentage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
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

def cols_to_dict(df:pd.DataFrame, a: str, b: str) -> dict :
    """
    Obtains a dictionary based on correspondence between two columns, for instance country names and ISO3 codes
    
    Parameters
    ----------
    df : pd.DataFrame
        the dataframe to start from
    a : str
        the column for dictionary keys
    b : str
        the column for dictionary values
    
    Returns
    -------
    dict
        dictionary of corrispondence of values in 'a' to values in 'b'
    """
    return df.set_index(a)[b].to_dict()

is_bijective: lambda x: len(set(x.keys())) == len(set(x.values()))  # whether dictionary is bijective/reversible

def keys_same_val(dict_: dict) -> dict:
    """
    Finds all keys in a dictionary that have the same values. 
    Useful to check if a dictionary can be inverted, or make it invertable.
    
    Parameters
    ----------
    dict_ : dict
        the dictionary to analyse.

    Returns
    -------
    to_return : dict
        {value: [list of keys with that value]}

    """
    s = pd.Series(dict_)
    dup_vals = list(set(s[s.duplicated()]))
    to_return = {}
    for dup in dup_vals:
        to_return[dup] = s[s == dup].index.tolist()
    return to_return