
# For getting the housing data
import os
import tarfile
from matplotlib.colors import hexColorPattern
from numpy.lib.function_base import median
from numpy.lib.histograms import _histogram_bin_edges_dispatcher
# from six.moves import urllib


# for reading the csv file
import pandas as pd

#For plotting the graph
import matplotlib.pyplot as plt

# for generating random numbers
import numpy as np
from sklearn.model_selection import train_test_split




'''
#When we want to get the housing data


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
 if not os.path.isdir(housing_path):
    os.makedirs(housing_path)
 tgz_path = os.path.join(housing_path, "housing.tgz")
 urllib.request.urlretrieve(housing_url, tgz_path)
 housing_tgz = tarfile.open(tgz_path)
 housing_tgz.extractall(path=housing_path)
 housing_tgz.close()

fetch_housing_data()

 '''


















#Ploting the Data

housing_data=pd.read_csv("datasets\housing\housing.csv")



# print(housing_data.head())
# print(housing_data.info())   # Gives information about the csv file
# print(housing_data["ocean_proximity"].value_counts())      #GIves the value_count
# print(housing_data.describe()) #Gives information of max min etc , attributes whose values are integers



'''


housing_data.hist(bins=50)  # Here bins is how many parts the histogram to be divided into  or simple bins are the no of bars
plt.show()



'''

# housing_data["median_income"].hist()



housing_data["income_cat"]=pd.cut(housing_data["median_income"],bins=[0,1.5,3,4.5,6.,np.inf],labels=[1,2,3,4,5]) # np.inf mean inserting positive infinity , -np.inf means negative infinity
# housing_data["income_cat"].hist()
# plt.show()

# we are making a new column named income_cat in housing_data DataFrame which contains the different categories of median_income
# i.e we are diving the median_income into 5 categories , if median_income is from 0 to 1.5 it is named a , 1.5 to 3.0 named b , so on 6 to np.inf i.e 6 to positive infinity named as e .
# housing_data["income_cat"]=pd.cut(housing_data["median_income"],bins=[0,1.5,3,4.5,6.,np.inf],labels=["a","b","c","d","e"])
# print(housing_data)
# print(housing_data["income_cat"].hist())
# plt.show()



# housing_data["income_cat"]=pd.cut(housing_data["median_income"],bins=[0,1.5,3,4.5,6.,np.inf],labels=[5,10,15,20,25])
# housing_data["income_cat"].hist()
# plt.show()



# housing_data["income_cat"]=pd.cut(housing_data["median_income"],bins=[0,1.5,3,4.5,6.,np.inf],labels=[100,200,300,400,500])
# housing_data["income_cat"].hist()
# plt.show()















"""
# Creating a test set on own code


def split_training_set(dataset,test_ratio):
      np.random.seed(45)            #This is used to generate the same ordered random numbers every time
      shuffled_indices=np.random.permutation(len(dataset))
      test_set_size=int(len(dataset)*test_ratio)
      test_set_indices=shuffled_indices[:test_set_size]
      train_set_indices=shuffled_indices[test_set_size:]
      return dataset.iloc[train_set_indices],dataset.iloc[test_set_indices]
   
train_set,test_set=split_training_set(housing_data,0.2)
print(test_set)
print(train_set)
"""



# Making train_set and test set using the predefined function available from sklearn.model_selection import train_test_split
# train_set,test_set=train_test_split(housing_data,test_size=0.2,random_state=40)
# print(train_set)
# print(test_set)



 
# Stratified split
from sklearn.model_selection import StratifiedShuffleSplit
a=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for  str_train_set_index,str_test_set_index in a.split(housing_data,housing_data["income_cat"]):
      str_train_set=housing_data.loc[str_train_set_index]
      # print(str_test_set_index)
      str_test_set=housing_data.loc[str_test_set_index]
#check
# print(str_test_set["income_cat"].value_counts()/len(str_test_set))

for boom in ( str_test_set,str_train_set):
      boom.drop("income_cat",axis=1,inplace=True)

#Making a copy ,so not to disturb the original data
housing=str_train_set.copy()

# housing.plot(kind="scatter" ,x="longitude", y="latitude")
# housing.plot(kind="scatter" ,x="longitude", y="latitude",alpha=0.1)
# plt.show()


#By default colour bar will be True
#s means the radius of bubbles may be in mm
# label means label for the bubble
# c means which property should determine colour
#we are using cmap template for colour
# housing.plot(kind="scatter",x="longitude", y="latitude" , s=housing["population"]/100 , label="population" , figsize=(10,7),c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True ) 
# plt.legend()
# plt.show()


# # # correlation
# corr_matrix=housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))


# # scatter matrix to get correlation graph
# from pandas.plotting import scatter_matrix
# attributes=["median_house_value","median_income", "total_rooms","housing_median_age"]
# scatter_matrix(housing[attributes],figsize=(12,8))
# plt.show()


## two plot the graph between the properties having the best correlation , i.e median_income and median_house_value so we can observe the straight lines in the data , so we can remove the data that gives the staright line which confuses the algorithm .
# housing.plot(kind="scatter" , x="median_income" , y="median_house_value" , alpha=0.1)
# plt.show()


# #creating new data using the pre existing data 
# housing["rooms_per_household"]=housing["total_rooms"]/housing["households"]
# housing["population_per_household"]=housing["population"]/housing["households"]
# housing["bedrooms_per_room"]=housing["total_bedrooms"]/housing["total_rooms"]

# corr_matrix=housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))





################################## Preparing data for machine learning algorithm ##################################

#lets create a new data set for cleaning 
#lets create or change the housing data set and give it the data set of stratified data set with out median_house_value
#lets take median_house_value as label
housing=str_train_set.drop("median_house_value",axis=1)  #This will copy the data of str_train_set to housing by removing median_house_value # axis = 1 indicates to remove the y axis
housing_label=str_train_set["median_house_value"].copy()
# print(housing_label)

# print(housing.info())# from this we can observe that the total_bedrooms value is missing in some rows

# #now we have 3 options
# #1) Remove the total_bedroom attribute
# housing.drop("total_bedroom",axis=1,inplace=True)
# #2) Remove the rows that are empty
# housing.dropna(subset=["total_bedrooms"],inplace=True)
# #3) To fill the empty cells with the median value or mean or zero value
# median=housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median,inplace=True)



#Imputer method to fill the cells that are empty
from sklearn.impute import SimpleImputer

housing_num=housing.drop("ocean_proximity",axis=1)
# print(housing.info())
# print(housing_num.info())

imputer=SimpleImputer(strategy="median")
imputer.fit(housing_num)
# print(imputer.statistics_) #All the median values are stored in imputer.statistics_
x=imputer.transform(housing_num)
## print(x)
##now we have x in the form of an numpy array and with no columns lets transform it back into dataframe
housing_tr=pd.DataFrame(x,columns=housing_num.columns)
# print(housing_tr.info()) #Now check , we can observe that all the cells are full


## Handling Text and Categorial Attributes
housing_cat=housing[["ocean_proximity"]]
# print(housing_cat.head(10))


from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder=OrdinalEncoder()
# housing_cat_encoder=ordinal_encoder.fit_transform(housing_cat)
# print(housing_cat_encoder[:10])
# print(ordinal_encoder.categories_)

#One HOt encoder

from sklearn.preprocessing import OneHotEncoder
cat_encoder=OneHotEncoder()
# housing_cat_1hot=cat_encoder.fit_transform(housing_cat) #Now this will be in SciPy sparse matrix.
# print(housing_cat_1hot)
# housing_cat_1hot=housing_cat_1hot.toarray()
# print(housing_cat_1hot)

"""
"""
from sklearn.base import BaseEstimator,TransformerMixin
rooms_ix,bedrooms_ix,population_ix,households_ix=3,4,5,6   #These are the indexes of the specified columns in the data frame or csv

class CombinedAtrributesAdder(BaseEstimator,TransformerMixin):
      def __init__(self,add_bedrooms_per_room=True):
            self.add_bedrooms_per_room=add_bedrooms_per_room
      def fit(self,X,y=None):
            return self
      def transform(self,X,y=None):
            rooms_per_household=X[:,rooms_ix]/X[:,households_ix]   #we are slicing a multi dimensional array , here we are saying to take all the values in first dimension that is all rows and we are specifying second dimension after comma , i.e here we are saying which column to select
            population_per_household=X[:,population_ix]/X[:,households_ix]
            if self.add_bedrooms_per_room:
                  bedrooms_per_room=X[:,bedrooms_ix]/X[:,rooms_ix]
                  return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]  #np.c_ will concatenate all the lists given inside it in side by side manner
            else:
                  return np.c_[X,rooms_per_household,population_per_household]

attr_adder=CombinedAtrributesAdder()
housing_extra_attribs=attr_adder.transform(housing.values)
# print(housing_extra_attribs)
# print(type(housing.values)) #It is a numpy array

#Transformation pipelines 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline=Pipeline([
      ("imputer",SimpleImputer(strategy="median")),
      ("attribs_adder",CombinedAtrributesAdder()),
      ("std_scalar",StandardScaler())
])
housing_num_tr=num_pipeline.fit_transform(housing_num)
# print(list(housing_num))   # This gives the list of indexes of housing num

#column transformer i.e it will transform both the categorial and the numerical columns in a single pipeline and will again combine it
from sklearn.compose import ColumnTransformer

num_attributes=list(housing_num)
cat_attributes=["ocean_proximity"]
full_pipeline=ColumnTransformer([
      ("num",num_pipeline,num_attributes),
      ("Car",OneHotEncoder(),cat_attributes)
])
housing_prepared=full_pipeline.fit_transform(housing)
# print(housing_prepared)


print(housing_prepared)


#select and train a model
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_label)

# some_data=housing.iloc[:5]
# # print(f"some_data = {some_data}")
# some_labels=housing_label.iloc[:5]
# # print(f"some labels = {some_labels}")
# # print(f"some labels = {list(some_labels)}")
# some_data_prepared=full_pipeline.transform(some_data)
# # print(f"some_data = {some_data_prepared}")

# # print("predictions: ",lin_reg.predict(some_data_prepared))



# from sklearn.metrics import mean_squared_error
# housing_predictions=lin_reg.predict(housing_prepared)
# lin_mse=mean_squared_error(housing_label,housing_predictions)
# lin_rmse=np.sqrt(lin_mse)
# print("lin_rmse",lin_rmse)



