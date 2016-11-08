import kNN
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
from pylab import *  
data,label=kNN.file2matrix('datingTestSet2.txt')
fig=plt.figure(1)
ax=fig.add_subplot(211)
ax.scatter(data[:,0],data[:,1],15*array(label),15*array(label))
xlabel('fly km')
ylabel('play game')


#fig=plt.figure(1)
ax=fig.add_subplot(212)
ax.scatter(data[:,1],data[:,2],15*array(label),15*array(label))
xlabel('play game')
ylabel('consume')
plt.show()
