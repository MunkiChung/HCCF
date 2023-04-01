import tensorflow as tf
import numpy as np

paramId = 0
biasDefault = False
params = {}
regParams = {}
ita = 0.2
leaky = 0.1

def getParamId():
	global paramId
	paramId += 1
	return paramId

def setIta(ITA):
	ita = ITA

def setBiasDefault(val):
	global biasDefault
	biasDefault = val

def getParam(name):
	return params[name]

def addReg(name, param):
	global regParams
	if name not in regParams:
		regParams[name] = param
	else:
		print('ERROR: Parameter already exists')

def addParam(name, param):
	global params
	if name not in params:
		params[name] = param

def defineRandomNameParam(shape, dtype=tf.compat.v1.float32, reg=False, initializer='xavier', trainable=True):
	name = 'defaultParamName%d'%getParamId()
	return defineParam(name, shape, dtype, reg, initializer, trainable)

def defineParam(name, shape, dtype=tf.compat.v1.float32, reg=False, initializer='xavier', trainable=True):
	global params
	global regParams
	assert name not in params, 'name %s already exists' % name
	if initializer == 'xavier':
		ret = tf.compat.v1.get_variable(name=name, dtype=dtype, shape=shape,
			initializer=tf.keras.initializers.GlorotNormal(),
			trainable=trainable)
	elif initializer == 'trunc_normal':
		ret = tf.compat.v1.get_variable(name=name, initializer=tf.compat.v1.random.truncated_normal(shape=[int(shape[0]), shape[1]], mean=0.0, stddev=0.03, dtype=dtype))
	elif initializer == 'zeros':
		ret = tf.compat.v1.get_variable(name=name, dtype=dtype,
			initializer=tf.compat.v1.zeros(shape=shape, dtype=tf.compat.v1.float32),
			trainable=trainable)
	elif initializer == 'ones':
		ret = tf.compat.v1.get_variable(name=name, dtype=dtype, initializer=tf.compat.v1.ones(shape=shape, dtype=tf.compat.v1.float32), trainable=trainable)
	elif not isinstance(initializer, str):
		ret = tf.compat.v1.get_variable(name=name, dtype=dtype,
			initializer=initializer, trainable=trainable)
	else:
		print('ERROR: Unrecognized initializer')
		exit()
	params[name] = ret
	if reg:
		regParams[name] = ret
	return ret

def getOrDefineParam(name, shape, dtype=tf.compat.v1.float32, reg=False, initializer='xavier', trainable=True, reuse=False):
	global params
	global regParams
	if name in params:
		assert reuse, 'Reusing Param %s Not Specified' % name
		if reg and name not in regParams:
			regParams[name] = params[name]
		return params[name]
	return defineParam(name, shape, dtype, reg, initializer, trainable)

def BN(inp, name=None):
	global ita
	dim = inp.get_shape()[1]
	name = 'defaultParamName%d'%getParamId()
	scale = tf.compat.v1.Variable(tf.compat.v1.ones([dim]))
	shift = tf.compat.v1.Variable(tf.compat.v1.zeros([dim]))
	fcMean, fcVar = tf.compat.v1.nn.moments(inp, axes=[0])
	ema = tf.compat.v1.train.ExponentialMovingAverage(decay=0.5)
	emaApplyOp = ema.apply([fcMean, fcVar])
	with tf.compat.v1.control_dependencies([emaApplyOp]):
		mean = tf.compat.v1.identity(fcMean)
		var = tf.compat.v1.identity(fcVar)
	ret = tf.compat.v1.nn.batch_normalization(inp, mean, var, shift,
		scale, 1e-8)
	return ret

def FC(inp, outDim, name=None, useBias=False, activation=None, reg=False, useBN=False, dropout=None, initializer='xavier', reuse=False, biasReg=False, biasInitializer='zeros'):
	global params
	global regParams
	global leaky
	inDim = inp.get_shape()[1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	W = getOrDefineParam(temName, [inDim, outDim], reg=reg, initializer=initializer, reuse=reuse)
	if dropout != None:
		ret = tf.compat.v1.nn.dropout(inp, rate=dropout) @ W
	else:
		ret = inp @ W
	if useBias:
		ret = Bias(ret, name=name, reuse=reuse, reg=biasReg, initializer=biasInitializer)
	if useBN:
		ret = BN(ret)
	if activation != None:
		ret = Activate(ret, activation)
	return ret

def Bias(data, name=None, reg=False, reuse=False, initializer='zeros'):
	inDim = data.get_shape()[-1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	temBiasName = temName + 'Bias'
	bias = getOrDefineParam(temBiasName, inDim, reg=False, initializer=initializer, reuse=reuse)
	if reg:
		regParams[temBiasName] = bias
	return data + bias

def ActivateHelp(data, method):
	if method == 'relu':
		ret = tf.compat.v1.nn.relu(data)
	elif method == 'sigmoid':
		ret = tf.compat.v1.nn.sigmoid(data)
	elif method == 'tanh':
		ret = tf.compat.v1.nn.tanh(data)
	elif method == 'softmax':
		ret = tf.compat.v1.nn.softmax(data, axis=-1)
	elif method == 'leakyRelu':
		ret = tf.compat.v1.maximum(leaky*data, data)
	elif method == 'twoWayLeakyRelu6':
		temMask = tf.compat.v1.to_float(tf.compat.v1.greater(data, 6.0))
		ret = temMask * (6 + leaky * (data - 6)) + (1 - temMask) * tf.compat.v1.maximum(leaky * data, data)
	elif method == '-1relu':
		ret = tf.compat.v1.maximum(-1.0, data)
	elif method == 'relu6':
		ret = tf.compat.v1.maximum(0.0, tf.compat.v1.minimum(6.0, data))
	elif method == 'relu3':
		ret = tf.compat.v1.maximum(0.0, tf.compat.v1.minimum(3.0, data))
	else:
		raise Exception('Error Activation Function')
	return ret

def Activate(data, method, useBN=False):
	global leaky
	if useBN:
		ret = BN(data)
	else:
		ret = data
	ret = ActivateHelp(ret, method)
	return ret

def Regularize(names=None, method='L2'):
	ret = 0
	if method == 'L1':
		if names != None:
			for name in names:
				ret += tf.compat.v1.reduce_sum(tf.compat.v1.abs(getParam(name)))
		else:
			for name in regParams:
				ret += tf.compat.v1.reduce_sum(tf.compat.v1.abs(regParams[name]))
	elif method == 'L2':
		if names != None:
			for name in names:
				ret += tf.compat.v1.reduce_sum(tf.compat.v1.square(getParam(name)))
		else:
			for name in regParams:
				ret += tf.compat.v1.reduce_sum(tf.compat.v1.square(regParams[name]))
	return ret

def Dropout(data, rate):
	if rate == None:
		return data
	else:
		return tf.compat.v1.nn.dropout(data, rate=rate)

def selfAttention(localReps, number, inpDim, numHeads):
	Q = defineRandomNameParam([inpDim, inpDim], reg=True)
	K = defineRandomNameParam([inpDim, inpDim], reg=True)
	V = defineRandomNameParam([inpDim, inpDim], reg=True)
	rspReps = tf.compat.v1.reshape(tf.compat.v1.stack(localReps, axis=1), [-1, inpDim])
	q = tf.compat.v1.reshape(rspReps @ Q, [-1, number, 1, numHeads, inpDim//numHeads])
	k = tf.compat.v1.reshape(rspReps @ K, [-1, 1, number, numHeads, inpDim//numHeads])
	v = tf.compat.v1.reshape(rspReps @ V, [-1, 1, number, numHeads, inpDim//numHeads])
	att = tf.compat.v1.nn.softmax(tf.compat.v1.reduce_sum(q * k, axis=-1, keepdims=True) / tf.compat.v1.sqrt(inpDim/numHeads), axis=2)
	attval = tf.compat.v1.reshape(tf.compat.v1.reduce_sum(att * v, axis=2), [-1, number, inpDim])
	rets = [None] * number
	paramId = 'dfltP%d' % getParamId()
	for i in range(number):
		tem1 = tf.compat.v1.reshape(tf.compat.v1.slice(attval, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
		# tem2 = FC(tem1, inpDim, useBias=True, name=paramId+'_1', reg=True, activation='relu', reuse=True) + localReps[i]
		rets[i] = tem1 + localReps[i]
	return rets

def lightSelfAttention(localReps, number, inpDim, numHeads):
	Q = defineRandomNameParam([inpDim, inpDim], reg=True)
	rspReps = tf.compat.v1.reshape(tf.compat.v1.stack(localReps, axis=1), [-1, inpDim])
	tem = rspReps @ Q
	q = tf.compat.v1.reshape(tem, [-1, number, 1, numHeads, inpDim//numHeads])
	k = tf.compat.v1.reshape(tem, [-1, 1, number, numHeads, inpDim//numHeads])
	v = tf.compat.v1.reshape(rspReps, [-1, 1, number, numHeads, inpDim//numHeads])
	# att = tf.compat.v1.nn.softmax(tf.compat.v1.reduce_sum(q * k, axis=-1, keepdims=True) * tf.compat.v1.sqrt(inpDim/numHeads), axis=2)
	att = tf.compat.v1.nn.softmax(tf.compat.v1.reduce_sum(q * k, axis=-1, keepdims=True) / tf.compat.v1.sqrt(inpDim/numHeads), axis=2)
	attval = tf.compat.v1.reshape(tf.compat.v1.reduce_sum(att * v, axis=2), [-1, number, inpDim])
	rets = [None] * number
	paramId = 'dfltP%d' % getParamId()
	for i in range(number):
		tem1 = tf.compat.v1.reshape(tf.compat.v1.slice(attval, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
		# tem2 = FC(tem1, inpDim, useBias=True, name=paramId+'_1', reg=True, activation='relu', reuse=True) + localReps[i]
		rets[i] = tem1 + localReps[i]
	return rets#, tf.compat.v1.squeeze(att)