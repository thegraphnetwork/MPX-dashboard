#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:14:23 2022

@author: nr
"""
import pendulum as pnd
import pandas as pd
from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator
from mpx_dashboard.functions import load_cases, group_and_aggr#, add_to_parquet, cached_read_csv, cached_read_parquet
from mpx_dashboard import csv_specs

@dag(
    # schedule=pd.to_timedelta('1 day'),
    schedule="@daily",
    # start_date=pnd.to_datetime('2022-10-13'),
    start_date=pd.to_datetime('2022-10-13'),
    catchup=False,
    tags=['ETL_MPX'],
)
def etl_mpx():
    # @task()
    def extract(rawfile: str):
        data = load_cases(rawfile, usecols=None, skiprows=None)
        # saving copy?
        return data
    
    @task(multiple_outputs=True)
    def transform(data: pd.DataFrame):
        if not len(data):
            raise ValueError('No data to aggregate')
        d_aggr = {}
        sus = data[data[csv_specs.statuscol] == csv_specs.statusvals['suspected']]
        d_aggr['cases'] = group_and_aggr(data, column=csv_specs.countrycol, date_col=csv_specs.confdatecol,
                            dropna=True, entrytype='cases', dropzeros=True)
        
        d_aggr['sus_cases'] = group_and_aggr(sus, column=csv_specs.countrycol, date_col=csv_specs.entrydatecol,
                            dropna=True, entrytype='cases', dropzeros=True)
        d_aggr['deaths'] = group_and_aggr(data, column=csv_specs.countrycol, date_col=csv_specs.deathdatecol,
                            dropna=True, entrytype='deaths', dropzeros=True)
        d_aggr['sus_deaths'] = group_and_aggr(sus[sus[csv_specs.outcomecol] == csv_specs.outcomevals['death']], column=csv_specs.countrycol,
                                             date_col=csv_specs.deathdatecol, # NB. there are some deaths with no deathdate. 
                                             # right now they are being excluded. We could plot them with the date of entry or of last modification
                                             dropna=True, entrytype='deaths', dropzeros=True)
        return d_aggr
    
    @task()
    def load(d_aggr: dict):
        for k, v in d_aggr.items():
            v.to_parquet(f'{k}.parquet')
    rawfile = 'https://raw.githubusercontent.com/globaldothealth/monkeypox/main/latest_deprecated.csv'
    data = PythonOperator(task_id='extract', python_callable=extract, op_args=[rawfile])
    data
    # data = extract(rawfile)
    # d_aggr = transform(data)
    # load(d_aggr)
    
etl_mpx()


