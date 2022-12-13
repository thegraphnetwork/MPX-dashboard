#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 17:59:41 2022

@author: nr
"""
from typing import Optional, Union#, Any
import streamlit as st
import pandas as pd
from mpx_dashboard import csv_specs 
import pyarrow.parquet as pq
import pyarrow as pa
from mpx_dashboard import utils
from epigraphhub.settings import env
from epigraphhub.connection import get_engine
import wbgapi as wb

@st.cache(allow_output_mutation=True)
def get_demographics_egh() -> pd.DataFrame:
    """

    Returns
    -------
    pd.Dataframe
        the demographics table from the google API

    """
    return pd.read_sql_table('demographics', 
                             get_engine(env.db.default_credential),
                             schema = 'google_health')

def get_country_pop_egh(loclist: Optional[list] = None, type_: str = 'country') -> pd.Series:
    """
    Gets the population for all countries required, from google API
    
    Parameters
    ----------
    loclist : Optional[list], optional
        the list of locations desired. The default is None.
    type_ : str, optional
        the type of location specifications. Default is 'country'. 
        Options are 'country' (case insensitive) anything containing 'iso3' (case insensitive), 
        otherwise 'location_key' is supposed. For countries, location_key is the iso2 code.

    Returns
    -------
    pd.Series
        the total population for each country in loclist, index is as loclist, according to type_,
        from the google API

    """
    pop = get_demographics_egh()
    if loclist:
        corr = False
        if type_.lower() == 'country':
            corr = {utils.country_to_location_key[i]: i for i in loclist}
            loclist = list(corr.keys())
        elif 'iso3' in type_.lower():
            corr = utils.iso2_to_iso3
            loclist = [utils.iso3_to_iso2[i] for i in loclist]
        pop = pop[pop['location_key'].isin(loclist)]
    else:
        corr = utils.location_key_to_country if type_.lower() == 'country' else utils.iso3_to_iso2 if 'iso2' in type_.lower() else False
    pop[type_] = pop['location_key'].map(corr) if corr else pop['location_key']
    return pop.set_index(type_)['population']

@st.cache(allow_output_mutation=True)
def get_wb_data(time: str, ind: Union[str, list] = ["SP.POP.TOTL.FE.IN", "SP.POP.TOTL.MA.IN"],
                country: Union[list, str] = 'all') -> pd.DataFrame:
    """
    Parameters
    ----------
    time : str
        time specification (e.g. YR2021).
    ind : Union[str, list], optional
        Field(s) required. The default is ["SP.POP.TOTL.FE.IN", "SP.POP.TOTL.MA.IN"].
    country : Union[list, str], optional
        ISO3 of the country/ies desired. The default is 'all'.

    Returns
    -------
    pd.DataFrame
        the requested table from the world bank.

    """
    return wb.data.DataFrame(series=ind, economy=country, db=2, time=time)
    
def get_country_pop_wb(year: Optional[int] = None,
                    loclist: Optional[list] = None, type_: str = 'country') -> pd.Series:
    """
    Parameters
    ----------
    year : int, optional
        The year that you want to extract the data. Default is None and uses most recent available
    loclist : list, optional
        the list of locations desired
    type_ : str
        the type of location specifications. Default is 'country'. 
        Options are 'country' (case insensitive) anything containing 'iso2' (case insensitive), 
        otherwise 'iso3' is supposed
    Returns
    -------
    pd.Series
        the total population for each country in loclist, index is as loclist, according to type_,
        from the world bank.
    """
    if year == None:
        year = pd.to_datetime('today').year

    ind = ["SP.POP.TOTL.FE.IN", "SP.POP.TOTL.MA.IN"] 
    if loclist:
        corr = False
        if type_.lower() == 'country':
            corr = {utils.country_to_iso3[i]: i for i in loclist}
            loclist = list(corr.keys())
        elif 'iso2' in type_.lower():
            corr = utils.iso3_to_iso2
            loclist = [utils.iso2_to_iso3[i] for i in loclist]
    else:
        corr = utils.iso3_to_country if type_.lower() == 'country' else utils.iso3_to_iso2 if 'iso2' in type_.lower() else False
    while True:
        try:
            time = f'YR{year}' # specify the year that you want to get 
            df = get_wb_data(ind, time, country=loclist)
            break
        except wb.APIResponseError: 
            year -= 1
    df = df.reset_index()
    df = df.rename(columns = {'economy': 'country', 'SP.POP.TOTL.FE.IN': 'total_female_pop', 'SP.POP.TOTL.MA.IN': 'total_male_pop'})
    df[type_] = df['country'].map(corr) if corr else df['country']
    df.set_index(type_, inplace = True)
    return df['total_female_pop'] + df['total_male_pop']

@st.cache(allow_output_mutation=True)
def load_cases(file: str, skiprows: Optional[int] = None, nrows: Optional[int] = None,
               usecols: Optional[list] = None, maptobool: Optional[list] = None, **kwargs) -> pd.DataFrame:
    """
    Reads disaggregated (case by case) data, dealing with type.
    
    Parameters
    ----------
    file : str
        the csv to read
    skiprows : int, optional
        the number of rows to skip
    nrows : int, optional
        number of rows to load. The default is None, i.e. all rows.
    usecols : list, optional
        columns to read. The default is None, i.e. all rows.
    maptobool : list, optional
        columns to map to a "pseudoboolean" (True, False, NaN)
    kwargs : dict
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

def add_to_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Appends to parquet file
    
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
    
@st.cache(allow_output_mutation=True)
def cached_read_csv(file: str, **kwargs) -> pd.DataFrame:
    """
    cached version of pd.read_csv

    Parameters
    ----------
    file : str
        file for pd.read_csv.
    **kwargs : dict 
        any argument for pd.read_csv

    Returns
    -------
    pd.DataFrame
        the content of the csv file
    """
    return pd.read_csv(file, **kwargs)

@st.cache(allow_output_mutation=True)
def cached_read_parquet(file, **kwargs) -> pd.DataFrame:
    """
    cached version of pd.read_parquet

    Parameters
    ----------
    file : str
        file for pd.read_parquet.
    **kwargs : dict 
        any argument for pd.read_parquet

    Returns
    -------
    pd.DataFrame
        the content of the parquet file
    """
    return pd.read_parquet(file, **kwargs)