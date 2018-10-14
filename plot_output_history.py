#!/usr/bin/python
import re
import sys
import json
import numpy as np
import math
import json


# importing the required module 
import matplotlib.pyplot as plt 

OUTPUT_HISTORY_FILE_NAME = 'output_history.txt'
  
f = open(OUTPUT_HISTORY_FILE_NAME, "r")

data = json.load(f)

f.close()

x = []
y1 = []
y2 = []

for entry in data:
	x.append(entry['epoch'])
	y1.append(entry['loss'])
	y2.append(entry['validation_loss'])
  
# plotting the points  
plt.plot(x, y1) 
plt.plot(x, y2)
  
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 
  
# giving a title to my graph 
plt.title('My first graph!') 
  
# function to show the plot 
plt.show() 