import numpy as np
import matplotlib.pyplot as plt

adam = np.genfromtxt("adam.txt")
adagrad = np.genfromtxt("adagrad.txt")
rmsprop = np.genfromtxt("rmsprop.txt")
spalera = np.genfromtxt("spalera.txt")
sgd = np.genfromtxt("sgd.txt")
smorms3 = np.genfromtxt("smorms3.txt")

a = np.linspace(0, 5, 315)

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.weight': 'light'})

plt.semilogy(a, adam, 'r-')
plt.semilogy(a, adagrad, 'b-')
plt.semilogy(a, rmsprop, 'g-')
plt.semilogy(a, spalera, 'k-')
plt.semilogy(a, smorms3, 'c-')
plt.semilogy(a, sgd, 'y-')

plt.legend(["Adam", "AdaGrad", "RMSprop", "SPALeRA", "SMORMS3", "SGD"])
plt.ylabel("loss")
plt.xlabel("epochs")

plt.savefig('learning_curves_crop.eps', format='eps', dpi=1000)

plt.show()