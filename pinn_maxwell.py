import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

R = 1

num_dense_layers = 9
num_dense_nodes = 494
lr = 0.009

def pde(x, y):
    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, i=1, j=1)

    return du_xx + du_yy

geom = dde.geometry.Disk([0, 0], R)

def boundary(x, on_boundary):
    return on_boundary

def boundary_top(x, on_boundary):
    return on_boundary and np.allclose(x, [0, 1])

def boundary_bottom(x, on_boundary):
    return on_boundary and np.allclose(x, [0. -1])

def boundary_left_right(x, on_boundary):
    return on_boundary and not np.allclose(x, [0, 1]) and not np.allclose(x, [0, -1])

def boundary_center(x, on_boundary):
    return on_boundary and np.allclose(x, [0, 0])

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

def func_t_b(x):
    normal = geom.boundary_normal(x)
    normal = np.array([normal])
    normal = np.sum(normal)
    return normal

def func_r_l(x):
    normal = geom.boundary_normal(x)
    normal = np.array([normal])
    normal = np.sum(0 * normal)
    return normal





bc_top = dde.icbc.NeumannBC(geom, func_t_b, boundary_top)
bc_bottom = dde.icbc.NeumannBC(geom, func_t_b, boundary_bottom)
bc_round = dde.icbc.NeumannBC(geom, func_r_l, boundary_left_right)

ic = dde.icbc.PointSetBC([0, 0], 0)


data = dde.data.PDE(geom, pde, [ic, bc_top, bc_bottom, bc_round], num_domain=6000, num_boundary=1000, num_test=3000)

layer_size = [2] + [num_dense_nodes] * num_dense_layers + [1]
activation = "sigmoid"
net = dde.nn.FNN(layer_size, activation, "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=lr)
losshistory, train_state = model.train(iterations=10000)

dde.saveplot(losshistory, train_state, issave=False, isplot=True)


Nx = 500
Ny = 500

xmin, xmax, ymin, ymax = [-1, 1, -1, 1]
plot_gird = np.mgrid[xmin : xmax : Nx * 1j, ymin : ymax : Ny * 1j]
points = np.vstack(
    (plot_gird[0].ravel(), plot_gird[1].ravel(), np.zeros(plot_gird[0].size))
)

points_2d = points[:2,:]
u = model.predict(points[:2,:].T)
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
