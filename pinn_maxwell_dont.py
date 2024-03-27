import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

R = 1
num_dense_layers = 5
num_dense_nodes = 150
activation = "tanh"

geom = dde.geometry.Disk([0, 0], R)

center = np.array([0, 0])

def my_degree(y, x, theta, input):
    phi = np.arctan2(y, x)
    first = phi - theta/2
    second = phi + theta/2
    x_min = min([np.cos(first), np.cos(second), np.cos(phi)])
    x_max = max([np.cos(first), np.cos(second), np.cos(phi)])
    y_min = min([np.sin(first), np.sin(second), np.sin(phi)])
    y_max = max([np.sin(first), np.sin(second), np.sin(phi)])
    return (x_min <= input[0] <= x_max) and (y_min <= input[1] <= y_max)

def pde(x, y):
    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, i=1, j=1)
    return du_xx + du_yy


def boundary_bottom(x, on_boundary):
    return on_boundary and my_degree(np.sin(3*np.pi/4), np.cos(3*np.pi/4), np.pi/8, x)
#(np.cos(9*np.pi/16) <= x[0] <= np.cos(7*np.pi/16)) and x[1] <= -np.sin(7*np.pi/16)
#my_degree(np.sin(3*np.pi/4), np.cos(3*np.pi/4), np.pi/8, x)

def boundary_top(x, on_boundary):
    return on_boundary and my_degree(np.sin(np.pi/2), np.cos(np.pi/2), np.pi/8, x)
#my_degree(np.sin(np.pi/2), np.cos(np.pi/2), np.pi/8, x)

def boundary_other1(x, on_boundary):
    return on_boundary and ((np.cos(9*np.pi/16) >= x[0] >= -1) or (np.cos(7*np.pi/16) <= x[0] <= 1))
#((np.cos(9*np.pi/16) >= x[0] >= -1) or (np.cos(7*np.pi/16) <= x[0] <= 1))
bc_c = dde.icbc.PointSetBC(center, 0, component=0)
bc_b = dde.icbc.DirichletBC(geom, lambda x : -1, boundary_bottom)
bc_t = dde.icbc.DirichletBC(geom, lambda x : 1, boundary_top)
bc_o1 = dde.icbc.NeumannBC(geom, lambda x : 0, boundary_other1)
 
data = dde.data.PDE(
    geom, pde, [bc_t, bc_b, bc_o1], num_domain=3000, num_boundary=150,
)

layer_size = [2] + [num_dense_nodes] * num_dense_layers + [1]
net = dde.nn.pytorch.fnn.FNN(layer_size, activation,"Glorot uniform")

model = dde.Model(data, net)
model.compile("adam", lr=1e-3) 
losshistory, train_state = model.train(iterations=15000)
dde.saveplot(losshistory,train_state,issave=False,isplot=True)

"""
Nx = 500
Ny = 500

xmin, xmax, ymin, ymax = [-1, 1, -1, 1]
plot_grid = np.mgrid[xmin : xmax : Nx * 1j, ymin : ymax : Ny * 1j]
points = np.vstack(
    (plot_grid[0].ravel(), plot_grid[1].ravel(), np.zeros(plot_grid[0].size))
)

points_2d = points[:2, :]
u = model.predict(points[:2, :].T)
u = u.reshape((Nx, Ny))

ide = np.sqrt(points_2d[0, :]**2 + points_2d[1,:]**2) > R
ide = ide.reshape((Nx, Ny))

plt.rc("font", family="serif", size=22)

fig, ax1 = plt.subplots(1, sharey=True, figsize=(24,12))

matrix = np.fliplr(u).T
matrix = np.ma.masked_where(ide, matrix)

pcm = ax1.imshow(
    matrix,
    extent=[-1,1,-1,1],
    cmap=plt.cm.get_cmap("seismic"),
    interpolation="spline16",
    label="PINN"
)

fig.colorbar(pcm, ax=ax1)

ax1.set_title("PINNs")

plt.savefig("plot_manufactured.pdf")
"""
