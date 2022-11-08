import pandas as pd
x=pd.read_csv("housing.csv")
# print(x.to_string())   
print(x.loc[0:1,"latitude"])