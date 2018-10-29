#!/usr/bin/python
import re
import sys
import json
import numpy as np
import math
import sys
program_name = sys.argv[0]
arguments = sys.argv[1:]

class_number = 3
FULL = False
MAX_EPOCHS = 50
if len(arguments) == 3:
	class_number = int(arguments[0])
	if(arguments[1]=="full"):
		FULL = True
	MAX_EPOCHS = int(arguments[2])
else:
	print "Need 3 arguments: 1st argument is class number (0-49), 2nd is full/verysmall, 3rd is max number of epochs"
	exit()

CLASSES_FILE_NAME = "classes.txt"

VOCAB_FILE_NAME = "vocab_verysmall.txt"
DATA_VECTORS_FILE_NAME = "data/vectors_sparse_train_verysmall.txt"	
VALID_DATA_VECTORS_FILE_NAME = "data/vectors_sparse_valid_verysmall.txt"
if FULL:
	VOCAB_FILE_NAME = "vocab_full.txt"
	DATA_VECTORS_FILE_NAME = "data/vectors_sparse_train_full.txt"
	VALID_DATA_VECTORS_FILE_NAME = "data/vectors_sparse_valid_full.txt"
OUTPUT_W_FILE_NAME = "output_multi/output_w"+str(class_number)+".txt"
OUTPUT_B_FILE_NAME = "output_multi/output_b"+str(class_number)+".txt"
OUTPUT_PARAMS_FILE_NAME = "output_multi/output_params"+str(class_number)+".txt"
OUTPUT_HISTORY_FILE_NAME = "output_multi/output_history"+str(class_number)+".txt"
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

VOCAB_SIZE = len(vocab)

def unpack_vector(sparse_vector, length):
	v = np.zeros((length,), dtype=np.float)
	for key, value in sparse_vector.items():
		key = int(key)
		value = float(value)
		v[key] = value
	return v

def sigma(z):
	value = 0.0
	try:
		value = 1.0/(1.0 + math.exp(-z))
	except OverflowError:
		print("Z value on overflow error was " + str(z))
		value = 0.0
	if (value == 1.0 or value == 0.0):
		print(str(class_number) + " " + str(z))
	return value

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
W = np.zeros((VOCAB_SIZE,), dtype=np.float)
b = 0
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

print("Data read, %d instances, W length %d. ALPHA = %f LAMBDA = %f MAX_EPOCHS = %d" %(NUM_INSTANCE_TO_PROCESS, VOCAB_SIZE, ALPHA, LAMBDA, MAX_EPOCHS))

output_history = []
J_VALID_PREVIOUS = 999999
# MAX_EPOCHS = 50
# if FULL:
# 	MAX_EPOCHS = 10
for epoch in range(0,MAX_EPOCHS):
	J = 0
	for i in range(0,NUM_INSTANCE_TO_PROCESS):
		# huge and problematic instance
		# if (i == 121004):
		# 	continue
		instance = data[i]
		x = instance['vector']
		y = 0
		if class_number in instance['classes']:
			y = 1
		z = sparse_mult(W, x) + b

		a = sigma(z)

		if ((a<=0) or (a >= 1)):
			# print("Instance no. " + str(i) + " caused issue")
			# print(data[i]['vector'])
			continue
		dz = a - y
		sparse_dW_times_ALPHA = sparse_scalar_mult(x, dz*ALPHA)
		db = dz
		# actual loss
		J += -(y*math.log(a) + (1-y)*math.log(1 - a)) + LAMBDA*np.dot(W,W)
		# update parameters
		# need to add 2*LAMBDA*wi to each wi for l2 regularization?
		W = sparse_subtract(W, sparse_dW_times_ALPHA)
		# W = (1.0 - LAMBDA)*W 
		# W = np.multiply(W, 1.0 - LAMBDA)
		b = b - ALPHA*db
	J = J/NUM_INSTANCE_TO_PROCESS

	# now calculate validation loss
	# iterate over all validation instances, calculate loss only
	J_VALID = 0
	for i in range(0, NUM_INSTANCE_TO_PROCESS_VALID):
		instance = valid_data[i]
		x = instance['vector']
		y = 0
		if class_number in instance['classes']:
			y = 1
		z = sparse_mult(W, x) + b
		a = sigma(z)
		if ((a<=0) or (a >= 1)):
			continue
		J_VALID += -(y*math.log(a) + (1-y)*math.log(1 - a)) + LAMBDA*np.dot(W,W)
	J_VALID = J_VALID/NUM_INSTANCE_TO_PROCESS_VALID

	print ("Class "+ str(class_number) + " Epoch " + str(epoch) + ", loss = " + str(J) + ", validation loss: "+ str(J_VALID) + ", W square = " + str(np.dot(W,W)))
	current_hist = {}
	current_hist['epoch'] = epoch
	current_hist['loss'] = J
	current_hist['validation_loss'] = J_VALID
	current_hist['w_square'] = str(np.dot(W,W))

	output_history.append(current_hist)

	f = open(OUTPUT_W_FILE_NAME, "w")
	f.write(json.dumps(W.tolist()))
	f.close()

	f = open(OUTPUT_B_FILE_NAME, "w")
	f.write(json.dumps(b))
	f.close()

	f = open(OUTPUT_PARAMS_FILE_NAME, "w")
	f.write("Epochs done: " + str(epoch + 1) + "\n")
	f.write("ALPHA: " + str(ALPHA) + "\n")
	f.write("LAMBDA: " + str(LAMBDA) + "\n")
	f.write("Number of instances: " + str(NUM_INSTANCE_TO_PROCESS) + "\n")
	f.write("Loss: " + str(J) + "\n")
	f.write("W squared: " + str(np.dot(W,W)) + "\n")
	f.close()

	f = open(OUTPUT_HISTORY_FILE_NAME, "w")
	f.write(json.dumps(output_history))
	f.close()
	# if J_VALID > J_VALID_PREVIOUS:
	# 	print("Validation loss increased. Exiting...")
	# 	break
	J_VALID_PREVIOUS = J_VALID
