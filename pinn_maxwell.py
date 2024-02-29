import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

R = 1
num_dense_layers = 5
num_dense_nodes = 150
activation = "tanh"

geom = dde.geometry.Disk([0.0, 0.0], R)

def pde(x, y):
    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, i=1, j=1)
    return du_xx + du_yy

def boundary_left(x, on_boudnary):
    return on_boudnary and x[0] <= 0

def boundary_right(x, on_boundary):
    return on_boundary and x[0] > 0

def Neumann_left(x):
    return -(1/np.sqrt(2 * np.pi)) * np.cos(x[:,1:2])

def Neumann_right(x):
    return (1/np.sqrt(2 * np.pi)) * np.cos(x[:,1:2])

bc_c = dde.icbc.PointSetBC(np.array([0.0, 0.0]), 0.0)

bc_neumann_left = dde.icbc.NeumannBC(
    geom,
    Neumann_left,
    boundary_left,
)

bc_neumann_right = dde.icbc.NeumannBC(
    geom,
    Neumann_right,
    boundary_right,
)

data = dde.data.PDE(
    geom, pde, [bc_c, bc_neumann_left, bc_neumann_right], num_domain=2540, num_boundary=80,
)

layer_size = [2] + [num_dense_nodes] * num_dense_layers + [1]
net = dde.nn.pytorch.fnn.FNN(layer_size, activation,"Glorot uniform")

def feature_transform(x):
    return torch.cat(
        [x[:, 0:1] * torch.sin(x[:,1:2]), x[:,0:1] * torch.cos(x[:,1:2])], dim=1
    )

net.apply_feature_transform(feature_transform)

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=10000)
dde.saveplot(losshistory,train_state,issave=False,isplot=True)

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