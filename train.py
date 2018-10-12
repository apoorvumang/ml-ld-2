#!/usr/bin/python
import re
import sys
import json
import numpy as np
import math

VOCAB_FILE_NAME = "vocab.txt"
CLASSES_FILE_NAME = "classes.txt"
DATA_VECTORS_FILE_NAME = "data/vectors_sparse.txt"
ALPHA = 0.01




vocab = {}
f = open(VOCAB_FILE_NAME, "r")
for line in f.readlines():
    line = line.strip()
    splitLine = line.split('\t', 2)
    word = splitLine[0]
    wordId = int(splitLine[1])
    vocab[word] = wordId
f.close()

VOCAB_SIZE = len(vocab)

def unpack_vector(sparse_vector, length):
	v = np.zeros((length,), dtype=np.float)
	for key, value in sparse_vector.items():
		key = int(key)
		value = float(value)
		v[key] = value
	return v

def sigma(z):
	return 1.0/(1.0 + math.exp(-z))

# parameters
W = np.zeros((VOCAB_SIZE,), dtype=np.float)
b = 0
vsigma = np.vectorize(sigma)

# sparse vector data with class
data = []
file = open(DATA_VECTORS_FILE_NAME, "r")

for line in file.readlines():
	line = json.loads(line)
	data.append(line)

for epoch in range(0,5):
	J = 0
	for i in range(0,1000):
		instance = data[i]
		x = unpack_vector(instance['vector'], VOCAB_SIZE)
		y = 0
		if 3 in instance['classes']:
			y = 1
		z = np.dot(W,x) + b
		# print(z)
		a = sigma(z)
		dz = a - y
		dW = x*dz
		db = dz
		# actual loss
		J += -(y*math.log(a) + (1-y)*math.log(1 - a))
		# update parameters
		W = W - ALPHA*dW
		b = b - ALPHA*db
	J = J/1000.0
	print ("Epoch done, loss = " + str(J))



print(len(data))
file.close()