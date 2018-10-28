#!/usr/bin/python
import re
import sys
import json
import numpy as np
import math
import json


# importing the required module 
import matplotlib.pyplot as plt 

OUTPUT_HISTORY_FOLDER = '../history/'
NUM_CLASSES = 50
MAX_EPOCHS = 20


x = []
y1 = []
y2 = []

for i in range(0, MAX_EPOCHS):
	x.append(i)
	y1.append(0)
	y2.append(0)

for i in range(0, NUM_CLASSES):
	OUTPUT_HISTORY_FILE_NAME =  OUTPUT_HISTORY_FOLDER + "output_history"+str(i)+".txt" 
	f = open(OUTPUT_HISTORY_FILE_NAME, "r")
	data = json.load(f)
	f.close()
	key = 0
	for entry in data:
		y1[key] += entry['loss']
		y2[key] += entry['validation_loss']
		key += 1


for i in range(0, MAX_EPOCHS):
	y1[i] = y1[i]/MAX_EPOCHS
	y2[i] = y2[i]/MAX_EPOCHS

# for entry in data:
# 	x.append(entry['epoch'])
# 	y1.append(entry['loss'])
# 	y2.append(entry['validation_loss'])
  
# plotting the points  
plt.plot(x, y1) 
plt.plot(x, y2)

# naming the x axis 
plt.xlabel('Epochs') 
# naming the y axis 
plt.ylabel('Loss') 

# giving a title to my graph 
plt.title('Test and Validation loss over epochs') 

# function to show the plot 
plt.show() 