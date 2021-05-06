# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 10:43:34 2021

@author: arda
"""

import csv
import matplotlib as plt
import seaborn
import pandas as pd


bankrupt = []
nonbankrupt = []
with open('data.csv', mode='r') as data:
    raw_data = pd.read_csv(data)
    print(raw_data.head())
    print(raw_data.at[0,'Bankrupt?'])
    
