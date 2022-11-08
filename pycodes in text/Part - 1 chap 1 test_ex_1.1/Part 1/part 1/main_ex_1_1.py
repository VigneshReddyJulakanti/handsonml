from operator import mod
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model

#Loading
gdp_per_cap=pd.read_csv("gdp_per_cap.csv",delimiter=",")
life_satis=pd.read_csv("life_satis.csv",delimiter=",")

full=pd.merge(right=life_satis,left=gdp_per_cap)
print(full)
# #data is ready 😮

full.plot(kind="scatter", y="gdp",x="ls" )
full.plot(kind="scatter", y="bonus",x="ls")


'''
#This is not used when plotting from a dataframe as there is a method to plot from the data frames which is better than this , i.e dataframe_name.plot(kind="scatter",x="",y="")
# plt.scatter(full["ls"],full["gdp"])
# plt.xlabel("ls")
# plt.ylabel("gdp")
'''

plt.show()


#selecting a linear model
model=sklearn.linear_model.LinearRegression()

#Train model
model.fit(gdp_per_cap[["gdp","bonus"]],life_satis["ls"])
    
# make prediction
print(model.predict([[50,5]]))