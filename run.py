#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File  : run.py.py
# @Author: nixin
# @Date  : 2020/11/4

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import minmax_scale
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from skcriteria import Data, MAX, MIN
from skcriteria.madm import simple, closeness
import plotly.graph_objects as go
import numpy as np

# pd.set_option('display.max_rows', 500)  
pd.set_option('display.max_columns', 500)    
pd.set_option('display.width', 4000)        



patent_data = pd.read_csv('data/patent_count_datails_priority_year.csv')
patent_data = patent_data[['patent_number', 'count_inventor_name', 'count_forward_cite_no_family',
                           'count_forward_cite_yes_family', 'count_backward_cite_no_family',
                           'count_backward_cite_yes_family', 'priority_date_year']].head(10)
print(patent_data)
print("=========")

#add similarity value feature to patent_data

patent_data[['similarity_value']] = pd.DataFrame(np.array([0.86,0.83,0.86,0.86]))
print(patent_data)
print("+++++++")


patent_data = patent_data.drop('priority_date_year', 1)
print(patent_data)
print(patent_data.iloc[:, 1:5])
print(patent_data.columns)
#
#
## project the goodness for each column
criteria_data = Data(patent_data.iloc[:, 1:7], [MAX, MAX, MAX, MAX,MAX,MAX],
                     anames= patent_data['patent_number'],
                     cnames= patent_data.columns[1:7],
                     weights= [0.1, 0.3, 0.1, 0.1, 0.1, 0.3]) ##assign weights to attributes
print(criteria_data)


def normalize_data(logic):
    df = patent_data.iloc[:, 1:6].values.copy()
    # print(df)
    if logic == "minmax":
        normalized_data = minmax_scale(df)
        # normalized_data[:, 6] = 1 - normalized_data[:, 6] ##for the min feature
    elif logic == "sumNorm":
        normalized_data = df / df.sum(axis=0)
        # normalized_data[:, 6] = 1 / normalized_data[:, 6]
    elif logic == "maxNorm":
        normalized_data = df / df.max(axis=0)
        # normalized_data[:, 6] = 1 / normalized_data[:, 6]
    return normalized_data



def plot_heatmap(logic):
    plot_datas = normalize_data(logic)
    patent_names = patent_data['patent_number']
    attribute_names = patent_data.columns[1:]
    sns.heatmap(plot_datas, annot=True, yticklabels = patent_names, xticklabels = attribute_names, fmt='.2g')

###########
# print final ranking table with different multi criteria decision makers

dm = simple.WeightedSum()
dec = dm.decide(criteria_data)
print(dec)
print(dec.e_.points) ##print each rank's value
print(dec.rank_) ##print ranks

print("==============================")

dm = simple.WeightedProduct()
dec = dm.decide(criteria_data)
print(dec)
print(dec.e_.points) ##print each rank's value
print(dec.rank_) ##print ranks

print("==============================")

dm = closeness.TOPSIS()
dec = dm.decide(criteria_data)
print(dec)
print("Ideal:", dec.e_.ideal) ##print each rank's value
print("Anti-Ideal:", dec.e_.anti_ideal)
print("Closeness:", dec.e_.closeness)

######################
#compartions with multiple solvers

patent_data_copy = patent_data.copy()

#weighted sum, sumNorm
dm = simple.WeightedSum(mnorm="sum")
dec = dm.decide(criteria_data)
patent_data_copy.loc[:, 'rank_weightedSum_sumNorm_inverse'] = dec.rank_

#weighted sum, maxNorm
dm = simple.WeightedSum(mnorm="max")
dec = dm.decide(criteria_data)
patent_data_copy.loc[:, 'rank_weightedSum_maxNorm_inverse'] = dec.rank_

#weighted product, sumNorm
dm = simple.WeightedProduct(mnorm="sum")
dec = dm.decide(criteria_data)
patent_data_copy.loc[:, 'rank_weightProduct_sumNorm_inverse'] = dec.rank_

#weighted product, maxNorm
dm = simple.WeightedProduct(mnorm="max")
dec = dm.decide(criteria_data)
patent_data_copy.loc[:, 'rank_weightedProduct_maxNorm_inverse'] = dec.rank_

#min max scale + mirror
patent_data_copy.loc[:, 'rank_weightedSum_minmaxScale_subtract'] =\
    pd.Series(normalize_data("minmax").sum(axis=1)).rank(ascending=False).astype(int)

#sort for better visualization
patent_data_copy.sort_values(by=['rank_weightedSum_maxNorm_inverse'], inplace=True)


