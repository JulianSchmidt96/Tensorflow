import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def f(x0, x1):
    return 100 * (x0**2 - x1)**2 + (x0 - 1)**2

def f_prime_x0(x0, x1):
    return 2 * (200 * x0 * (x0**2 - x1) + x0 - 1)

def f_prime_x1(x0, x1):
    return -200 * (x0**2 - x1)

def plot_rosenbrock(downhill=False, x0=-1, x1=-1):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca(projection='3d')

    s = 0.3
    X = np.arange(-2, 2.+s, s)
    Y = np.arange(-2, 3.+s, s)
        
    #Create the mesh grid(s) for all X/Y combos.
    X, Y = np.meshgrid(X, Y)
    #Rosenbrock function w/ two parameters using numpy Arrays
    Z = f(X, Y)

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, alpha=0.8)
    # Global minimum
    ax.scatter(1, 1, f(1, 1), color="red", marker="*", s=200)
    # Starting point
    ax.scatter(x0, x1, f(x0, x1), color="green", marker="o", s=200)

    if downhill:
        eps = 50
        # Plot Updated Points
        for (x0, x1) in downhill:
            ax.scatter(x0, x1, f(x0, x1)+eps, color="green", marker="o", s=50)
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


x0=np.random.uniform(-2,2)
x1=np.random.uniform(-2,2)

y=f(x0,x1)

print("\n\nGlobales Minimum bei: " , 1 , 1)

it=0
eta= 0.005
stop_iter=100000
downhill_points = []

while it < stop_iter:
    x0 = x0 - eta * f_prime_x0(x0, x1)
    x1 = x1 - eta * f_prime_x1(x0, x1)
    it += 1
    fx = f(x0, x1)
    if it %100==0:
        downhill_points.append([x0, x1])

print("Solutin ", fx)
print("x0= ", x0)
print("x1= ", x1)
plot_rosenbrock(downhill=downhill_points, x0=x0, x1=x1)