#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:22:46 2022

@author: nico
"""
import pandas as pd

datecols = []
isdatecol = lambda x: x.startswith('Date')
def iscat(c):
    c = c.lower()
    return True if True in ['country' in c, 'gender' in c, 'status' in c, 'outcome' in c, '(y/n/na)' in c] else False
# isstring  = lambda x: True if True in ['history' in x, 'comment' in x, 'source' in x] or x in ['Location', 'Genomics_Metadata', 'Symptoms'] else False
    
def guess_dtype(c):
    if iscat(c):
        return 'category'
    if isdatecol(c):
        return 'object'
    return pd.StringDtype()
    
ispseudobool = lambda x: '(Y/N/NA)' in x
pseudobooldict = {'Y': True, 'N': False, 'NA': None}
countrycol = 'Country'
confdatecol = 'Date_confirmation'
gendercol = 'Gender'
outcomecol = 'Outcome'
outcomevals = ["Death", "Recovered"]  # do not include Nan
hospitalcol = 'Hospitalised (Y/N/NA)'
entrydatecol = 'Date_entry'
lastmoddatecol = 'Date_last_modified'
deathdatecol = 'Date_death'