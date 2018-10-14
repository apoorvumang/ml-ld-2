#!/usr/bin/python
import re
import sys
import json
import numpy as np
import math

VOCAB_FILE_NAME = "vocab.txt"
CLASSES_FILE_NAME = "classes.txt"
DATA_VECTORS_FILE_NAME = "data/vectors_sparse.txt"
VALID_DATA_VECTORS_FILE_NAME = "data/vectors_sparse_valid.txt"
OUTPUT_W_FILE_NAME = "output_multi_w.txt"
OUTPUT_B_FILE_NAME = "output_multi_b.txt"
OUTPUT_PARAMS_FILE_NAME = "output_multi_params.txt"
OUTPUT_HISTORY_FILE_NAME = "output_multi_history.txt"
ALPHA = 0.005
LAMBDA = 0.00

vocab = {}
f = open(VOCAB_FILE_NAME, "r")
for line in f.readlines():
    line = line.strip()
    splitLine = line.split('\t', 2)
    word = splitLine[0]
    wordId = int(splitLine[1])
    vocab[word] = wordId
f.close()

classes = {}
f = open(CLASSES_FILE_NAME, "r")
for line in f.readlines():
    line = line.strip()
    splitLine = line.split('\t', 2)
    docClass = splitLine[0]
    classId = int(splitLine[1])
    classes[docClass] = classId
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
	if(z < -1000):
		print(z)
		return 0
	if(z > 1000):
		print(z)
		return 1
	return 1.0/(1.0 + math.exp(-z))

def sparse_mult(normal, sparse):
	ans = 0.0
	for key, value in sparse.items():
		key = int(key)
		value = float(value)
		ans = ans + normal[key]*value
	return ans

def sparse_scalar_mult(sparse, k):
	sparse2 = sparse.copy()
	for key in sparse2:
		sparse2[key] *= k
	return sparse2

def sparse_subtract(normal, sparse):
	for key, value in sparse.items():
		key = int(key)
		value = float(value)
		normal[key] -= value
	return normal

# parameters
W = []
b = []
for i in range(len(classes)):
	W.append(np.zeros((VOCAB_SIZE,), dtype=np.float))
	b.append(0)

vsigma = np.vectorize(sigma)

# sparse vector data with class
data = []
file = open(DATA_VECTORS_FILE_NAME, "r")
NUM_INSTANCE_TO_PROCESS = 3000000 # if this exceeds no. of lines in file its still ok
count = 0
for line in file.readlines():
	line = json.loads(line)
	data.append(line)
	count = count + 1
	if(count > NUM_INSTANCE_TO_PROCESS):
		break
NUM_INSTANCE_TO_PROCESS = count
file.close()

# validation data
valid_data = []
file = open(VALID_DATA_VECTORS_FILE_NAME, "r")
NUM_INSTANCE_TO_PROCESS_VALID = 3000000 # if this exceeds no. of lines in file its still ok
count = 0
for line in file.readlines():
	line = json.loads(line)
	valid_data.append(line)
	count = count + 1
	if(count > NUM_INSTANCE_TO_PROCESS_VALID):
		break
NUM_INSTANCE_TO_PROCESS_VALID = count
file.close()


output_history = []
for epoch in range(0,20):
	J = [0]*len(classes)
	for i in range(0,NUM_INSTANCE_TO_PROCESS):
		instance = data[i]
		x = instance['vector']
		for j in range(len(classes)):
			curr_W = W[j]
			curr_b = b[j]
			curr_J = J[j]
			y = 0
			if j in instance['classes']:
				y = 1
			z = sparse_mult(curr_W, x) + curr_b
			a = sigma(z)
			if ((a<=0) or (a >= 1)):
				# print("Instance no. " + str(i) + " caused issue")
				# print(data[i]['vector'])
				continue
			dz = a - y
			sparse_dW_times_ALPHA = sparse_scalar_mult(x, dz*ALPHA)
			db = dz
			# actual loss
			curr_J += -(y*math.log(a) + (1-y)*math.log(1 - a)) + LAMBDA*np.dot(curr_W,curr_W)
			# update parameters
			# need to add 2*LAMBDA*wi to each wi for l2 regularization?
			curr_W = sparse_subtract(curr_W, sparse_dW_times_ALPHA)
			# W = (1.0 - LAMBDA)*W 
			# W = np.multiply(W, 1.0 - LAMBDA)
			curr_b = curr_b - ALPHA*db
	J = [k / NUM_INSTANCE_TO_PROCESS for k in J]
	# now calculate validation loss
	# iterate over all validation instances, calculate loss only
	J_VALID = [0]*len(classes)
	for i in range(0, NUM_INSTANCE_TO_PROCESS_VALID):
		instance = valid_data[i]
		x = instance['vector']
		for j in range(len(classes)):
			curr_W = W[j]
			curr_b = b[j]
			curr_J_VALID = J_VALID[j]
			y = 0
			if j in instance['classes']:
				y = 1
			z = sparse_mult(curr_W, x) + curr_b
			a = sigma(z)
			curr_J_VALID += -(y*math.log(a) + (1-y)*math.log(1 - a)) + LAMBDA*np.dot(curr_W,curr_W)
	J_VALID = [k / NUM_INSTANCE_TO_PROCESS_VALID for k in J_VALID]

	print("Epochs done: " + str(epoch+1))
	
	current_hist = {}
	current_hist['epoch'] = epoch
	current_hist['loss'] = J
	current_hist['validation_loss'] = J_VALID
	# current_hist['w_square'] = str(np.dot(W,W))

	output_history.append(current_hist)

	f = open(OUTPUT_W_FILE_NAME, "w")
	new_W = []
	for w in W:
		new_W.append(w.tolist())
	f.write(json.dumps(new_W))
	f.close()

	f = open(OUTPUT_B_FILE_NAME, "w")
	f.write(json.dumps(b))
	f.close()

	f = open(OUTPUT_PARAMS_FILE_NAME, "w")
	f.write("Epochs done: " + str(epoch + 1) + "\n")
	f.write("ALPHA: " + str(ALPHA) + "\n")
	f.write("LAMBDA: " + str(LAMBDA) + "\n")
	f.write("Number of instances: " + str(NUM_INSTANCE_TO_PROCESS) + "\n")
	f.close()

	f = open(OUTPUT_HISTORY_FILE_NAME, "w")
	f.write(json.dumps(output_history))
	f.close()
print(len(data))
