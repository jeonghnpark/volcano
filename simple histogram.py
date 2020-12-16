import matplotlib.pyplot as plt
import numpy as np
np.random.seed(19680801)

mu=100
sigma=15
x=mu+sigma*np.random.randn(437)

num_bins=50

fig, (ax, ax2)=plt.subplots(1,2)

n,bins,patches=ax.hist(x,num_bins, density=True)
n2,bins2,patches2=ax2.hist(x,num_bins, density=False)

y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
y2 = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins2 - mu))**2))

ax.plot(bins,y,'--')
ax2.plot(bins2,y2,'--')


plt.show()