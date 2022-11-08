import matplotlib.pyplot as plt
import numpy as np


###
a=[1,2,3,4,5,6,7]
b=[2,3,4,5,6,7,8]
# plt.plot(a,b)  #plot(a,b) means join the given two points with a line
###






###
'''To plot a histogram'''
a=[1,2,3,4,5,6,7,4,2,25,7,2,33,3,6,3,5,33,3,63,63,63,6,3,63,62,1,2,2,7,8,2,14,26,8,26,26,3,26,3,74,8,2,3,7,2,6,32,7,4,2,74,37,2,3,48,34,8,2,7,2,63,2,62,7,2,3,2,3,3,2,3,9]

plt.hist(a,bins=50)   # Here bind is how many parts the histogram to be divided into  or simple bins are the no of bars
###






plt.show()