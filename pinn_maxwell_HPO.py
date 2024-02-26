import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

np.int = int

R = 1
iterations = 10000

def pde(x, y):
    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, i=1, j=1)
    return du_xx + du_yy

def boundary(x, on_boundary):
    return on_boundary

def boundary_top(x, on_boundary):
    return on_boundary and np.allclose(x, [0, 1])

def boundary_bottom(x, on_boundary):
    return on_boundary and np.allclose(x, [0, -1])

def boundary_left_right(x, on_boundary):
    return on_boundary and not boundary_top(x, on_boundary) and not boundary_bottom(x, on_boundary)

geom = dde.geometry.Disk([0, 0], R)

def func(x):
    
    if np.allclose(x, [0, 1]):
        normal = geom.boundary_normal(x)
        normal = np.array([normal])
        result = np.sum(normal)
    elif np.allclose(x, [0, -1]):
        normal = geom.boundary_normal(x)
        normal = np.array([normal])
        result = np.sum(normal)
    else:
        normal = geom.boundary_normal(x)
        normal = np.array([normal])
        result = np.sum(0 * normal)
    return result

def creat_model(config):
    
    learning_rate, num_dense_layers, num_dense_nodes, activation = config

    
    bc_round = dde.icbc.NeumannBC(geom, func, boundary)
    ic = dde.icbc.PointSetBC(np.array([0,0]), np.array([0]))

    nx_train = int(6000)
    nx_test = int(3000)

    data = dde.data.PDE(
        geom,
        pde,
        [ic, bc_round],
        num_domain=nx_train,
        num_boundary=int(nx_train/6),
        num_test=nx_test,
    )

    net = dde.maps.FNN(
        [2] + [num_dense_nodes]*num_dense_layers + [1],
        activation,
        "Glorot uniform",
    )

    model = dde.Model(data, net)
    model.compile("adam", lr=learning_rate)
    return model

def train_model(model, config):
    losshistory, tarin_state = model.train(iterations=iterations)
    train = np.array(losshistory.loss_train).sum(axis=1).ravel()
    test = np.array(losshistory.loss_test).sum(axis=1).ravel()

    error = test.min()
    return error

n_calls = 50
dim_learning_rate = Real(low=1e-4, high=5e-2, name="learning_rate", prior="log-uniform")
dim_num_dense_layers = Integer(low=1, high=10, name="num_dense_layers")
dim_num_dense_nodes = Integer(low=5, high=500, name="num_dense_nodes")
dim_activation = Categorical(categories=["sin", "sigmoid", "tanh"], name="activation")

dimensions = [
    dim_learning_rate,
    dim_num_dense_layers,
    dim_num_dense_nodes,
    dim_activation,
]

default_parameters = [1e-3, 4, 50, "tanh"]

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation):
  config = [learning_rate, num_dense_layers, num_dense_nodes, activation]
  global ITERATION

  print(ITERATION, "it number")
  print("learning rate : {0:.1e}".format(learning_rate))
  print("num_dense_layers:", num_dense_layers)
  print("num_dense_nodes:", num_dense_nodes)
  print("activation:", activation)
  print()

  model = creat_model(config)
  error = train_model(model, config)

  if np.isnan(error):
    error = 10**5

  ITERATION += 1
  return error

ITERATION = 0

search_result = gp_minimize(
    func=fitness,
    dimensions=dimensions,
    acq_func="EI",  # Expected Improvement.
    n_calls=n_calls,
    x0=default_parameters,
    random_state=1234,
)

print(search_result.x)

plot_convergence(search_result)
plot_objective(search_result, show_points=True, size=3.8)