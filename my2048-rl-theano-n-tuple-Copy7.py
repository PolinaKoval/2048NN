from env2048.env2048 import Game2048
from c2048 import push

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import DenseLayer, InputLayer
from lasagne.layers import FlattenLayer, ConcatLayer

floatX = theano.config.floatX

from lasagne.layers.dnn import Conv2DDNNLayer


def Winit(shape):
	rtn = np.random.normal(size=shape).astype(floatX)
	rtn[np.random.uniform(size=shape) < 0.9] *= 0.01
	return rtn


input_var = T.tensor4()
target_var = T.vector()
N_FILTERS = 512
N_FILTERS2 = 4096

_ = InputLayer(shape=(None, 16, 4, 4), input_var=input_var)

conv_a = Conv2DDNNLayer(_, N_FILTERS, (2, 1), pad='valid')  # , W=Winit((N_FILTERS, 16, 2, 1)))
conv_b = Conv2DDNNLayer(_, N_FILTERS, (1, 2), pad='valid')  # , W=Winit((N_FILTERS, 16, 1, 2)))

conv_aa = Conv2DDNNLayer(conv_a, N_FILTERS2, (2, 1), pad='valid')  # , W=Winit((N_FILTERS2, N_FILTERS, 2, 1)))
conv_ab = Conv2DDNNLayer(conv_a, N_FILTERS2, (1, 2), pad='valid')  # , W=Winit((N_FILTERS2, N_FILTERS, 1, 2)))

conv_ba = Conv2DDNNLayer(conv_b, N_FILTERS2, (2, 1), pad='valid')  # , W=Winit((N_FILTERS2, N_FILTERS, 2, 1)))
conv_bb = Conv2DDNNLayer(conv_b, N_FILTERS2, (1, 2), pad='valid')  # , W=Winit((N_FILTERS2, N_FILTERS, 1, 2)))

_ = ConcatLayer([FlattenLayer(x) for x in [conv_aa, conv_ab, conv_ba, conv_bb, conv_a, conv_b]])
l_out = DenseLayer(_, num_units=1, nonlinearity=None)

prediction = lasagne.layers.get_output(l_out)
P = theano.function([input_var], prediction)
loss = lasagne.objectives.squared_error(prediction, target_var).mean() / 2
accuracy = lasagne.objectives.squared_error(prediction, target_var).mean()
params = lasagne.layers.get_all_params(l_out, trainable=True)
updates = lasagne.updates.adam(loss, params, beta1=0.5)


train_fn = theano.function([input_var, target_var], loss, updates=updates)
loss_fn = theano.function([input_var, target_var], loss)
accuracy_fn = theano.function([input_var, target_var], accuracy)


table = {2 ** i: i for i in range(1, 16)}
table[0] = 0

def make_input(grid):
	g0 = grid
	r = np.zeros(shape=(16, 4, 4), dtype=floatX)
	for i in range(4):
		for j in range(4):
			v = g0[i, j]
			r[table[v], i, j] = 1
	return r

def Vchange(grid, v):
	g0 = grid
	g1 = g0[:, ::-1, :]
	g2 = g0[:, :, ::-1]
	g3 = g2[:, ::-1, :]
	r0 = grid.swapaxes(1, 2)
	r1 = r0[:, ::-1, :]
	r2 = r0[:, :, ::-1]
	r3 = r2[:, ::-1, :]
	xtrain = np.array([g0, g1, g2, g3, r0, r1, r2, r3], dtype=floatX)
	ytrain = np.array([v] * 8, dtype=floatX)
	train_fn(xtrain, ytrain)

arrow = [3, 0, 1, 2]

def gen_sample_and_learn(game):
	game_len = 0
	game_score = 0
	last_grid = None
	while True:
		grid_array = game.grid.get_values()
		board_list = []
		for m in range(4):
			g = grid_array.copy()
			s = push(g, m % 4)
			if s >= 0:
				board_list.append((g, m, s))
		if board_list:
			boards = np.array([make_input(g) for g, m, s in board_list], dtype=floatX)
			p = P(boards).flatten()
			game_len += 1
			best_move = -1
			best_v = None
			for i, (g, m, s) in enumerate(board_list):
				v = 2 * s + p[i]
				if best_v is None or v > best_v:
					best_v = v
					best_move = m
					best_score = 2 * s
					best_grid = boards[i]
			game.step(arrow[best_move])
			game_score += best_score
		else:
			best_v = 0
			best_grid = None
		if last_grid is not None:
			Vchange(last_grid, best_v)
		last_grid = best_grid
		if not board_list:
			break
	return game_len, grid_array.max(), game_score


results = []
game = Game2048(4)
for j in range(200):
	game.reset()
	result = gen_sample_and_learn(game)
	print(j, result)
	results.append(result)
	dots_data = [[], [], []]
	if result[1] >= 2048:
		break
