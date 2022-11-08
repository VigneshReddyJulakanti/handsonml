import pandas as pd


# if headers are not present in csv : then put header = None
# x=pd.read_csv('testcsvpanda.csv',header=None)
# print(x)
# output:
# #header will be as#
# 0 1 2 3 4 5 6 ..


# To add a prefix to the above one , we use prefix="name_to_act_as_prefix"
# x=pd.read_csv('testcsvpanda.csv',header=None,prefix="hlo")
# print(x)
# output:
# #header will be as#
# hlo0 hlo1 hlo2 hlo3 hlo4


# x=pd.read_csv('testcsvpanda.csv',thousands=',',delimiter="\t")
# print(x)
# delimeter is the sequence of words used to separate cells in csv
# thousands=',' --> This means inplace of thousands we use .s

######################################


housing_data=pd.read_csv("housing.csv")



# print(len(housing_data))  # will print the len of the dataFrame

###################################


# print(housing_data.info())   # Gives information about the csv file
'''
output:
RangeIndex: 20640 entries, 0 to 20639
Data columns (total 10 columns):     
 #   Column              Non-Null Count  Dtype
---  ------              --------------  -----
 0   longitude           20640 non-null  float64
 1   latitude            20640 non-null  float64
 2   housing_median_age  20640 non-null  float64
 3   total_rooms         20640 non-null  float64
 4   total_bedrooms      20433 non-null  float64
 5   population          20640 non-null  float64
 6   households          20640 non-null  float64
 7   median_income       20640 non-null  float64
 8   median_house_value  20640 non-null  float64
 9   ocean_proximity     20640 non-null  object
dtypes: float64(9), object(1)
memory usage: 1.6+ MB
None
'''








# print(housing_data["ocean_proximity"].value_counts())      #GIves the value_count

'''
output:
<1H OCEAN     9136
INLAND        6551
NEAR OCEAN    2658
NEAR BAY      2290
ISLAND           5
Name: ocean_proximity, dtype: int64
'''






# print(housing_data.describe()) #Gives information of max min etc , attributes whose values are integers
"""
output:
<1H OCEAN     9136
INLAND        6551
NEAR OCEAN    2658
NEAR BAY      2290
ISLAND           5
Name: ocean_proximity, dtype: int64
          longitude      latitude  housing_median_age  ...    households  median_income  median_house_value
count  20640.000000  20640.000000        20640.000000  ...  20640.000000   20640.000000        20640.000000    
mean    -119.569704     35.631861           28.639486  ...    499.539680       3.870671       206855.816909    
std        2.003532      2.135952           12.585558  ...    382.329753       1.899822       115395.615874    
min     -124.350000     32.540000            1.000000  ...      1.000000       0.499900        14999.000000    
25%     -121.800000     33.930000           18.000000  ...    280.000000       2.563400       119600.000000    
50%     -118.490000     34.260000           29.000000  ...    409.000000       3.534800       179700.000000    
75%     -118.010000     37.710000           37.000000  ...    605.000000       4.743250       264725.000000    
max     -114.310000     41.950000           52.000000  ...   6082.000000      15.000100       500001.000000    

[8 rows x 9 columns]
"""
