from numpy import row_stack
import pandas as pd
import matplotlib as plt
import sys


###
#Note: Put 'delimeter="\t"' as parameter in  every read_csv()

'''
### imp notes ###

! series --> Single dimension , it is like a single column in a table.
! DataFrame --> Multi dimensional , its a table.

'''
###
 
'''


### series ###

#series
# a=[8,5,9]
# x=pd.Series(a)
# print(x)
# output
# 0    8
# 1    5
# 2    9


#index or label
# a=[8,5,9]
# x=pd.Series(a,index=["a","h","p"])
# print(x)
# output
# a    8      
# h    5      
# p    9      
# dtype: int64



# #refer by label
# a=[8,4,7]
# x=pd.Series(a,index=["i","t","k"])
# print(x["t"])
# output
# 4

# dictionary to series
# calories = {"day1": 420, "day2": 380, "day3": 390}
# myvar = pd.Series(calories)
# print(myvar)
# output
# day1    420
# day2    380
# day3    390
# dtype: int64

#specific items to include in series
# import pandas as pd
# calories = {"day1": 420, "day2": 380, "day3": 390}
# myvar = pd.Series(calories, index = ["day1", "day2"])
# print(myvar)
# output
# day1    420
# day2    380
# dtype: int64

'''
###

'''

### DataFrames ###

# hlo ={
#     'cars':['a','b','c'],
#     'cost':[8,9,3]
# }
# x=pd.DataFrame(hlo)
# print(x)
# output
#   cars  cost
# 0    a     8
# 1    b     9
# 2    c     3

#locate row
# hlo ={
#     'cars':['a','b','c'],
#     'cost':[8,9,3]
# }
# x=pd.DataFrame(hlo)
# print(x.loc[0])    # for printing multiple rows use something like loc[0,1] or like loc[0:]
# output
# cars    a
# cost    8
# Name: 0, dtype: object


#Giving index
# data = {
#   "calories": [420, 380, 390],
#   "duration": [50, 40, 45]
# }
# df = pd.DataFrame(data, index = ["day1", "day2", "day3"])
# print(df) 
# output
#       calories  duration
# day1       420        50
# day2       380        40
# day3       390        45

'''

###

'''

### Read csv ###


# x=pd.read_csv('testcsvpanda.csv')
# print(x)
# output
# gives the ouput in the form of DataFrame ( This will read the data seperated by comma in csv file)

# By default when we print a DataFrame then we will get only the first five rows and last five rows to get the total code we need to add .to_string()
# df = pd.read_csv('testcsvpanda.csv')
# print(df.to_string())


### Read JSON ###


# df = pd.read_json('data.json')
# print(df.to_string()) 
#JSON file is similar to a dic in python

'''
'''

###

### Pandas analyzing Data ###

# printing first n rows and headers from starting
# x=pd.read_csv("testcsvpanda.csv")
# print(x.head(8))
# output
# DataFrame of headers and first 8 rows
# If the number is not specified in the parenthesis then by deafault it takes the number as 5


#x.tail() this will give the last 5 rows

'''
#info
# x=pd.read_csv("housing.csv")
# print(x.info())

# '''

###

# ''' 

### cleaning ###



#Empty cells 

# return a new DataFrame that do not contain any empty cell , by removing the rows with empty cells with out altering the previous One 
# x=pd.read_csv("testcsvpanda.csv")
# y=x.dropna()
# print(y.to_string())


# if we dont want a new DataFrame and want to change the orginal one then use inplace = True in dropna
# x=pd.read_csv("testcsvpanda.csv")
# x.dropna(inplace=True)
# print(x.to_string())

# subset
x=pd.read_csv("testcsvpanda.csv")
x.dropna(subset=["idd"], inplace=True)
print(x.to_string())

#drop a specific row
# x=pd.read_csv("testcsvpanda.csv")
# x.drop(5, inplace=True)   # this will drop the 5 th row
# print(x.to_string())

#drop a specific column , we use an extra parameter named axis by default it will be 0 , if it is 1 then it says remove column
# x=pd.read_csv("testcsvpanda.csv")
# x.drop("idd", inplace=True,axis=1)   # this will drop the idd column
# print(x.to_string())


# Replace null with some other value
# x=pd.read_csv("testcsvpanda.csv")
# x.fillna("hlooo",inplace=True)  
# print(x.to_string())

# replace null with some other value only of specified column
# x=pd.read_csv("testcsvpanda.csv")
# x["name"].fillna("hlooo",inplace=True)  
# print(x.to_string())


# mean() ( The sum of all the numbers divided by no.of values )
# x=pd.read_csv("testcsvpanda.csv")
# y=x["idd"].mean()
# x["idd"].fillna(y,inplace=True)
# print(x.to_string())
# we can also use median() mode()
#median() : the value in middle ,after you have sorted in ascending order.
#mode() : the value repeated the most times.

###

# Wrong format
# x=pd.read_csv("testcsvpanda.csv")
# x["idd"]=pd.to_datetime(x["idd"])
# print(x.to_string())
#Nat : Not a time 

###
  

# wrong data
# x=pd.read_csv("testcsvpanda.csv")


# Replace a value 
# x=pd.read_csv("testcsvpanda.csv")
# x.loc[3,"idd"]=45  # Replace a column idd of row 3 to 45
# print(x.to_string())


#looping and x.index
# for i in x.index:
#     if x.loc[i,"idd"]>10:
#         print("hai")

#duplicates
# x.drop_duplicates(inplace=True)
# print(x.to_string())

'''

'''

#################### x.corr() ###################
# This gives the correlation


# this gives the correlation between different columns 
# Note : this neglects the non integer rows 

# x=pd.read_csv("testcsvpanda.csv")
# print(x.corr())

# the values range from -1 to 1

# 1 symbolises there is a strong corelation between , if one increses other too increases
# -1 symbolises , if one increses other decreases 

# '''

###

'''


### plot ###

# x=pd.read_csv("testcsvpanda.csv")
# x.plot()
# plt.pyplot.show()


# x=pd.read_csv("testcsvpanda.csv")
# x.plot( x='name' , y='idd')
# plt.pyplot.show()


'''

# x=pd.read_csv("testcsvpanda.csv")
# x.dropna(subset=["idd"] , inplace=True)
# print(x.to_string())


