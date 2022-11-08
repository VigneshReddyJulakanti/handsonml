import pandas as pd


###
'''Merging two dataframes'''
#if we have two dataframes named a and b and if we want to merge them , then we use new_DataFrameName=pd.merge(right=a,left=b)
# new_DataFrameName=pd.merge(right=a,left=b)




#### .loc() vs .iloc() ####
# loc() uses label or integer | df.loc([A:B])  A and B both are inclusive | Used to get certain Rows
# iloc() uses integer  | df.iloc([n:m]) n is inclusive and m is exclusive | Used to get certain columns 





# #### Adding id to the DataFrame
# data_set_housing=pd.read_csv("housing.csv")
# print(data_set_housing)
# data_set_housing_with_id=data_set_housing.reset_index()      #This adds id to the data frame i.e index from 0 to len(df) and label the column as index
# print(data_set_housing_with_id)
# print(data_set_housing_with_id["index"])




'''Under construction
# data_set_housing=pd.read_csv("housing.csv")
# def check(a):
#     return a <3
# data_set_housing_with_id=data_set_housing.reset_index()      #This adds id to the data frame i.e index from 0 to len(df) and label the column as index

# sett=data_set_housing_with_id["index"].apply(lambda id_:check(id_))
# print(sett)
# print(type(sett))
# print(data_set_housing_with_id.loc[sett])
# print(data_set_housing_with_id.loc[~sett])
# # print(check(5))
'''



# '''
#Divide into categories
#If we want to divide and label certain column with labels 
# we are making a new column named income_cat in housing_data DataFrame which contains the different categories of median_income
# i.e we are diving the median_income into 5 categories , if median_income is from 0 to 1.5 it is named a , 1.5 to 3.0 named b , so on 6 to np.inf i.e 6 to positive infinity named as e .
import numpy as np
import matplotlib.pyplot as plt
data_set_housing=pd.read_csv("housing.csv")
data_set_housing["median_income_cat"]=pd.cut(data_set_housing["median_income"],bins=[0.,1.5,3.,4.5,6.,np.inf],labels=["a","b","c","d","e"])
# print(data_set_housing)
# data_set_housing["median_income_cat"].hist()
# plt.show()(

# '''

from sklearn.model_selection import StratifiedShuffleSplit
a=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for str_train_set_index , str_test_set_index in a.split(data_set_housing,data_set_housing["median_income_cat"]):
    str_train_set=data_set_housing.loc[str_train_set_index]
    str_test_set=data_set_housing.loc[str_test_set_index]

print(str_train_set["median_income_cat"].value_counts())
print(str_test_set["median_income_cat"].value_counts())