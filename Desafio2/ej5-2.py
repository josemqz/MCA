import numpy as np
from matplotlib import pyplot as plt

R = 1
N = 1000

theta = np.random.uniform(0,2*np.pi, N)
T = np.random.gamma(3,1, N)

X = R*T*np.cos(theta)
Y = R*T*np.sin(theta)

t = np.arange(0,N)

fig, ax = plt.subplots(2, figsize=(10,10), sharex=True)
# plt.figure(figsize=(15,10))
ax[0].scatter(X, Y)
# ax[1].plot(t, Y, 'r')
ax[1].hist(T)
# ax[0].setlabel('X')


plt.show()

