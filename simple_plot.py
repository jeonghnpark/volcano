# import tkinter
# import
import matplotlib.pyplot as plt
# import matplotlib
import numpy as np
# plt.ion()

t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)
fig, ax = plt.subplots(nrows=2, ncols=2)

# fig, ax = plt.subplots()

ax[0,1].plot(t, s)

ax[0,1].set(xlabel='time', ylabel='voltage', title='about as simple')
ax[0,1].grid()
fig.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show(block=False)

x1=np.linspace(0.0, 5.0)
x2=np.linspace(0.0, 2.0)

y1=np.cos(2*np.pi*x1)*np.exp(-x1)
y2=np.cos(2*np.pi*x2)

fig, (ax1, ax2)=plt.subplots(2,1)
fig.suptitle("A tale of 2 subplots")

ax1.plot(x1, y1, 'o-')
ax2.plot(x2, y2, '.-')

ax1.set_ylabel('Damped oscillation')
ax2.set_ylabel('Undamped')
plt.show()


