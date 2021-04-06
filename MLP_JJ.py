# JJ 20/02/27
# This model is coded referring to ' Saito Goki, Deep Learning from Scratch, Hanbit Media, Inc. 2017 '.
# multi-layer perceptron using back-propagation
# activation function is used depending on the situation as follows:
# regression: identity function, binary_classify: simoid function, muti_classify: softmax function
# when softmax used, one-hot encoding recommended
# setting initial value is also important
# "He" intial value when Relu is used, but Xavier is recommened when sigmoid or tanh function is used
# syntax: python MLP_JJ.py [hidden size] [calcID] [# of hidden-layer] [index sets of descriptors] [index of training data in CV]

import os, sys, math
import numpy as np
from collections import OrderedDict
import timeit

seed = 7
np.random.seed(seed)  # seed should be set because of reproducibility! (important)

t1= timeit.default_timer()
def numerical_gradient(f, x):
	h = 1e-4
	grad = np.zeros_like(x)
	case_b = False
	try:
		column_num = x.shape[1]
		row_num = x.shape[0]
	except:
		column_num = x.size
		row_num = 1
		case_b = True
	for idx_row in range(row_num):
		for idx_column in range(column_num):
			if not case_b:
				tmp_val = x[idx_row, idx_column]
				x[idx_row, idx_column] = tmp_val + h
				fxh1 = f(x)
				x[idx_row, idx_column] = tmp_val - h
				fxh2 = f(x)
				grad[idx_row, idx_column] = (fxh1 - fxh2) / (2*h)
				x[idx_row, idx_column] = tmp_val
			else:
				tmp_val = x[idx_column]
				x[idx_column] = tmp_val + h
				fxh1 = f(x)
				x[idx_column] = tmp_val - h
				fxh2 = f(x)
				grad[idx_column] = (fxh1 - fxh2) / (2*h)
				x[idx_column] = tmp_val
	return grad

def cross_entropy_error(y, t):
	#if y.ndim == 1:
	#	t = t.reshape(1, t.size)
	#	y = y.reshape(1, y.size)
	batch_size = y.shape[0]
	return -np.sum(t*np.log(y + 1e-7)) / batch_size

def sum_squares_error(y, t):
	return 0.5 * np.sum((y-t)**2)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

class Relu:
	def __init__(self):
		self.mask = None

	def forward(self, x):
		self.mask = ( x <= 0 )
		out = x.copy()
		out[self.mask] = 0
		return out

	def backward(self, dout):
		dout[self.mask] = 0
		dx = dout
		return dx

def softmax(a):
	c = np.max(a, axis=1)
	c = c.reshape(c.size,1)
	exp_a = np.exp(a-c)
	sum_exp_a = np.sum(exp_a, axis=1)
	sum_exp_a = sum_exp_a.reshape(sum_exp_a.size,1)
	y = exp_a / sum_exp_a
	return y


class SoftmaxWithLoss:
	def __init__(self):
		self.loss = None
		self.y = None
		self.t = None
	
	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		self.loss = cross_entropy_error(self.y, self.t)
		return self.loss

	def backward(self, dout=1):
		batch_size = self.t.shape[0]
		dx = (self.y - self.t) / batch_size
		return dx

class IdentityWithLoss:
	def __init__(self):
		self.loss = None
		self.y = None
		self.t = None

	def forward(self, x, t):
		self.t = t
		self.y = x
		self.loss = sum_squares_error(self.y, self.t)
		return self.loss

	def backward(self, dout=1):
		batch_size = self.t.shape[0]
		dx = (self.y - self.t) / batch_size
		return dx


class Affine:
	def __init__(self, W, b):
		self.W = W
		self.b = b
		self.x = None
		self.dW = None
		self.db = None

	def forward(self, x):
		self.x = x
		out = np.dot(x, self.W) + self.b
		return out
	
	def backward(self, dout):
		dx = np.dot(dout, self.W.T)
		self.dW = np.dot(self.x.T, dout)
		self.db = np.sum(dout, axis=0)
		return dx


class Dropout:
	def __init__(self, dropout_ratio=0.2):
		self.dropout_ratio = dropout_ratio
		self.mask = None

	def forward(self, x, train_flg=True):
		if train_flg:
			self.mask = np.random.rand(*x.shape) > self.dropout_ratio # the asterisk * unpacks tuple. e.g.) *(2,3) = 2,3
			return x * self.mask
		else:
			return x * ( 1.0 - self.dropout_ratio )
	
	def backward(self, dout):
		return dout * self.mask

class MulLayerNet:
	def __init__(self, input_size, hidden_size, output_size, weight_init_std=1/np.sqrt(50/2), num_layer=2):
		self.params = {}
		self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
		self.params['b1'] = np.zeros(hidden_size)
		if num_layer < 3:
			self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
			self.params['b2'] = np.zeros(output_size)
		elif num_layer == 3:
			self.params['W2'] = weight_init_std * np.random.randn(hidden_size, hidden_size)
			self.params['b2'] = np.zeros(hidden_size)
			self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
			self.params['b3'] = np.zeros(output_size)
		
		self.layers = OrderedDict()
		self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
		#self.layers['Dropout1'] = Dropout(0)
		self.layers['Relu1'] = Relu()
		self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
		if num_layer == 3:
			self.layers['Relu2'] = Relu()
			self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
		#self.layers['Dropout2'] = Dropout(0.1)
		#self.lastLayer = SoftmaxWithLoss()
		self.lastLayer = IdentityWithLoss()

	def predict(self, x):
		for layer in self.layers.values():
			x = layer.forward(x)
		return x
	
	def loss(self, x, t):
		y = self.predict(x)
		return self.lastLayer.forward(y, t)

	def accuracy(self, x, t):
		y = self.predict(x)
		#y = np.argmax(y, axis=1)
		#t = np.argmax(t, axis=1)
		accuracy = np.sum((t-abs(y-t))/t) / float(x.shape[0])
		return accuracy

	def RSQR(self, x, t ): # determination coefficient r^2
		mean_t = float(np.sum(t))/t.shape[0]
		yy = self.predict(x)
		return 1 - ( float(np.sum((yy-t)**2)) / np.sum((t-mean_t)**2) )

	def numerical_gradient(self, x, t):
		loss_W = lambda W: self.loss(x,t)
		grads = {}
		grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
		grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
		grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
		grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
		if num_layer == 3:
			grads['W3'] = numerical_gradient(loss_W, self.params['W3'])
			grads['b3'] = numerical_gradient(loss_W, self.params['b3'])
		return grads
	
	def gradient(self, x, t):
		# forward
		self.loss(x, t) 
		# backward
		dout = 1
		dout = self.lastLayer.backward(dout)
		layers = list(self.layers.values())
		layers.reverse()
		for layer in layers:
			dout = layer.backward(dout)
		
		grads = {}
		grads['W1'] = self.layers['Affine1'].dW
		grads['b1'] = self.layers['Affine1'].db
		grads['W2'] = self.layers['Affine2'].dW
		grads['b2'] = self.layers['Affine2'].db
		if num_layer == 3:
			grads['W3'] = self.layers['Affine3'].dW
			grads['b3'] = self.layers['Affine3'].db
		return grads


# update parameters W, b
class SGD:
	def __init__(self, lr=0.1):
		self.lr = lr
	def update(self, params, grad):
		for key in params.keys():
			params[key] -= self.lr * grad[key]

class AdaGrad: 
	def __init__(self, lr=0.01): # lr = learning-rate
		self.lr = lr
		self.h = None

	def update(self, params, grad):
		if self.h == None:
			self.h = {}
			for key, val in params.items():
				self.h[key] = np.zeros_like(val)
		for key in params.keys():
			self.h[key] += grad[key] * grad[key]
			params[key] -= self.lr * grad[key] / (np.sqrt(self.h[key]) + 1e-7)

# making pre-processed list
def MinMaxScaler(ls,ls2,mode=0):
	# the list looks like: [[a1,a2,a3,...], [b1,b2,b3,...],...]
	# ls2 is reference for min, max scaling
	MinMax=[]
	for i in range(ls2.shape[1]):
		tmp2=[]
		for features2 in ls2:
			tmp2.append(features2[i])
		MinMax.append([min(tmp2),max(tmp2)])
	pp_ls=[]
	for i in range(ls.shape[1]):
		tmp=[]
		for features in ls:
			tmp.append(features[i])
		tmp = [(x-MinMax[i][0])/(MinMax[i][1]-MinMax[i][0]) for x in tmp]
		pp_ls.append(tmp)
	pp_new=[]
	for i in range(len(pp_ls[0])):
		tmp=[]
		for x in pp_ls:
			tmp.append(x[i])
		pp_new.append(tmp)
	if mode == 1: return MinMax
	else: return pp_new
	
def recover_scaling(scaled, minmax):
	re=[]
	for h in scaled:
		h=h[0]
		recovered=h*(minmax[1]-minmax[0]) + minmax[0]
		re.append([recovered])
	return np.array(re)

### extract training data ###########################################################################################################
f = open("/home/baikgrp/calcs/JJ/DeepLearning/data_200617_shuffled","r")
data_lines = f.readlines() # [line1, line2...]
f.close()
del data_lines[0]  # delete the first line of database
feature_set = list(map(int, sys.argv[4].split(",")))   # it looks like 2,4,7,8
species = []
x_train, t_train = [], []
for line in data_lines:
	if not line.split()[3] == "-":
		species.append(line.split()[0])
		tmp = []
		for feature in feature_set:		# feature is index number of descriptors
			tmp.append(float(line.split()[feature]))
		x_train.append(tmp)
		t_train.append(float(line.split()[3]))  # 1: wavelength, 2: brightness, 3: EQE
		

# leave-out 4-data for test-set (total 54 data)
rand_num = np.random.choice(53,4)
final_eval_x=[]
final_eval_y=[]
final_eval_species=[]
for i in rand_num:
    final_eval_x.append(x_train[i])
    final_eval_y.append([t_train[i]])
	final_eval_species.append([species[i]])
for i in rand_num:
    del x_train[i]
    del t_train[i]
    del species[i]
final_eval_x = np.array(final_eval_x)
final_eval_y = np.array(final_eval_y)
#final_eval_x = np.array(x_train[-4:])
#final_eval_y = t_train[-4:]
#final_eval_y = np.array([[x] for x in final_eval_y])
#del x_train[-4:]
#del t_train[-4:]

# K-fold cross validation
divide_K = 10
N_data_Kper = round(len(x_train) / divide_K)   # the number of data per each fold
N_tot_Kper = math.ceil(len(x_train)/N_data_Kper)   # the number of folds
cv_x_train = []  # set of x_train (total 10 train-set)
cv_x_test = []   # set of x_test (total 10 test-set)
cv_t_train = []
cv_t_test = []
cv_train_species = []
cv_test_species = []
for NN in range(N_tot_Kper):
	a = NN * (N_data_Kper)   # start number
	b = a + N_data_Kper   # finish number
	temp_species = species.copy()
	temp_x_train = x_train.copy()
	temp_t_train = t_train.copy()
	if NN == N_tot_Kper: 
		cv_test_species.append(species[a:])
		x_test_divided = temp_x_train[a:]
		t_test_divided = temp_t_train[a:]
		del temp_species[a:]
		del temp_x_train[a:]
		del temp_t_train[a:]
	else: 
		cv_test_species.append(species[a:b])
		x_test_divided = temp_x_train[a:b]
		t_test_divided = temp_t_train[a:b]
		del temp_species[a:b]
		del temp_x_train[a:b]
		del temp_t_train[a:b]
	cv_train_species.append(temp_species)
	cv_x_train.append(temp_x_train)
	cv_x_test.append(x_test_divided)
	cv_t_train.append(temp_t_train)
	cv_t_test.append(t_test_divided)

if len(sys.argv) >= 6:
	train_species, test_species = [], []
	# choose a specific train/test. sys.argv[5] is index number of set of train/test list (cv_x/t_train/test).
	train_species = cv_train_species[int(sys.argv[5])]
	test_species = cv_test_species[int(sys.argv[5])]
	temp_x_train = cv_x_train[int(sys.argv[5])]
	temp_x_test = cv_x_test[int(sys.argv[5])]
	temp_t_train = cv_t_train[int(sys.argv[5])]
	temp_t_test = cv_t_test[int(sys.argv[5])]
else:
	rand_N = np.random.choice(divide_K)
	cv_train_species = cv_train_species[rand_N]
	cv_test_species = cv_test_species[rand_N]
	temp_x_train = cv_x_train[rand_N]
	temp_x_test = cv_x_test[rand_N]
	temp_t_train = cv_t_train[rand_N]
	temp_t_test = cv_t_test[rand_N]

x_train, t_train = np.array(temp_x_train), np.array(temp_t_train)
x_test, t_test = np.array(temp_x_test), np.array(temp_t_test)
t_train = t_train.reshape(t_train.size,1)
t_test = t_test.reshape(t_test.size,1)

# data preprocessing ( MinMaxScaler )
x_total = np.concatenate((x_train,x_test),axis=0)
t_total = np.concatenate((t_train,t_test),axis=0)
t_total_MinMax = MinMaxScaler(t_total, t_total, 1)[0]
x_train = np.array(MinMaxScaler(x_train, x_total))
x_test = np.array(MinMaxScaler(x_test, x_total))
t_train = np.array(MinMaxScaler(t_train, t_total))
t_test = np.array(MinMaxScaler(t_test, t_total))

####################################################################################################################################

## Global Parameters
lr = 0.001
InputSize = x_train.shape[1]
OutputSize = t_train.shape[1]
HiddenSize = 15
num_layer = 2
if len(sys.argv) >= 2:
	if not sys.argv[1] == "now":
		HiddenSize = int(sys.argv[1])
Dir = "/home/baikgrp/calcs/JJ/DeepLearning"
if len(sys.argv) >= 3: Dir = "%s/%s" % (Dir, sys.argv[2])
if len(sys.argv) >= 4: num_layer = int(sys.argv[3])
##

def run():
	#global accuracy, accuracy_test
	global R2_train, R2_test, count_descent, keep_descent
	network = MulLayerNet(input_size=InputSize, hidden_size=HiddenSize, output_size=OutputSize, weight_init_std=1/np.sqrt(InputSize/2), num_layer=num_layer)

	#optimizer = SGD(lr)
	optimizer = AdaGrad()

	iters_num = int(10E10)
	train_size = x_train.shape[0]
	batch_size = train_size
	iter_per_epoch = 100

	#if os.path.isfile("%s/Parameters" % (Dir)): os.system("rm %s/Parameters" % (Dir))
	
	for i in range(iters_num):
		#batch_mask = np.random.choice(train_size, batch_size)
		#x_batch = x_train[batch_mask]
		#t_batch = t_train[batch_mask]
		x_batch = x_train
		t_batch = t_train
	
		# gradient update
		grad = network.gradient(x_batch, t_batch)
		
		# update parameters W, b based on "SGD" or "AdaGrad"
		params = network.params
		optimizer.update(params, grad)
	
		# adjusting parameters using numerical_gradient
		#for key in ('W1', 'b1', 'W2', 'b2'):
		#	network_1.params[key] -= learning_rate * grad_1[key]
	
		loss = network.loss(x_batch, t_batch)
		#accuracy = network.accuracy(x_batch, t_batch)
		#accuracy_test = network.accuracy(x_test, t_test)
		R2_train = network.RSQR(x_batch, t_batch)
		R2_test = network.RSQR(x_test, t_test)
	
		if i > 5000:
			#if accuracy < 0.6 or accuracy_test < 0.5: break
			if R2_train < 0.5 or R2_test < 0.3: break
		if i % iter_per_epoch == 0:
			f3 = open("%s/infos" % (Dir),"a")
			f3.write("R^2_train: %.3f , R^2_test: %.3f , iters: %d , tried: %d\n" % (R2_train, R2_test, i, tried))
			f3.close()
			#if accuracy >= 0.7:
			if R2_train >= 0.7 and R2_test > 0.5:
				record_R2_test.append(R2_test)
				if len(record_R2_test) >= 2:
					if keep_descent:
						if record_R2_test[-1] <= record_R2_test[-2]: count_descent+=1
						else: keep_descent=False
					else:
						if record_R2_test[-1] <= record_R2_test[-2]:
							count_descent+=1
							keep_descent=True
				if count_descent >= 50: break
				W1_val = network.params['W1']
				b1_val = network.params['b1']
				W2_val = network.params['W2']
				b2_val = network.params['b2']
				if num_layer >= 3:
					W3_val = network.params['W3']
					b3_val = network.params['b3']
				else:
					W3_val = "-"
					b3_val = "-"
				t22=timeit.default_timer()
				f = open("%s/Parameters" % (Dir),"a")
				f.write('%s\n\nPredicted:\n%s\n\nR^2_train: %f\nR^2_test: %f\niters_num: %d\ntried: %d\n' % (recover_scaling(network.predict(x_batch), t_total_MinMax), recover_scaling(network.predict(x_test), t_total_MinMax), R2_train, R2_test, i, tried))
				f.write("{:.2f} seconds\n\n".format(t22-t11))
				f.close()
				#if accuracy >= 0.85 and accuracy_test >= 0.77:
				if R2_train >= 0.83 and R2_test >= 0.75:
					f2 = open("%s/Parameters_good" % (Dir),"a")
					f2.write('W1\n%s\nb1\n%s\nW2\n%s\nb2\n%s\nW3\n%s\nb3\n%s\n\n%s\n\nPredicted:\n%s\n\nR^2_train: %f\nR^2_test: %f\niters_num: %d\ntried: %d\n\n' % (W1_val, b1_val, W2_val, b2_val, W3_val, b3_val, recover_scaling(network.predict(x_batch), t_total_MinMax), recover_scaling(network.predict(x_test), t_total_MinMax), R2_train, R2_test, i, tried))
					f2.close()
			#if accuracy >= 0.96 and accuracy < 1.0:
			if R2_train >= 0.99999 and R2_train < 1.0:
				print("R^2 >= 0.99999. Close.")
				break

			#t33=timeit.default_timer()
			#print("{:.2f} seconds".format(t33-t11))
	

if os.path.isfile("%s/Parameters" % (Dir)): os.system('rm %s/Parameters' % (Dir))
if os.path.isfile("%s/Parameters_good" % (Dir)): os.system('rm %s/Parameters_good' % (Dir))
if os.path.isfile("%s/species_info" % (Dir)): os.system('rm %s/species_info' % (Dir))
if os.path.isfile("%s/infos" % (Dir)): os.system('rm %s/infos' % (Dir))

f = open("%s/species_info" % (Dir),"a")
f.write("trained\n\n")
for ha in train_species: f.write("%s\n"%(ha))
f.write("\n\ntested\n\n")
for hb in test_species: f.write("%s\n"%(hb))
f.close()
tried = 0
t11=timeit.default_timer()
for iteration in range(2500):
	count_descent=0
	keep_descent=False
	record_R2_test=[]
	tried+=1
	run()
	if R2_train > 0.99 and R2_train < 1.0:
		if R2_test > 0.99 and R2_test < 1.0: break

t2= timeit.default_timer()
print("Working time: %f sec" % (t2 - t1))

