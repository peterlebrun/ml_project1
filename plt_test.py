#import matplotlib.pyplot as p
#
#p.plot([5, 6, 7, 8], [1, 2, 3, 4], 'ro')
#p.ylabel('numbazz')
#p.xlabel('idfk')
#p.axis([0, 10, 0, 10])
#p.show()

#import numpy as n
#import matplotlib.pyplot as p
#
## evenly sampled time at 200ms intervals
#x = n.arange(0., 5., 0.2)
#y1 = x**2
#y2 = x**2 + 1
#
##p.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
#lines = p.plot(x, y1, x, y2)
#p.setp(lines, color='r', linewidth=2.0)
#
#p.show()

#import numpy as np
#import matplotlib.pyplot as plt
#
#def f(t):
#    return np.exp(-t) * np.cos(2*np.pi*t)
#
#t1 = np.arange(0.0, 5.0, 0.1)
#t2 = np.arange(0.0, 5.0, 0.02)
#
#plt.figure(1)
#plt.subplot(222) #numrows, numcols, figure_num
#plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
#
##plt.subplot(212)
##plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
#plt.show()

#import matplotlib.pyplot as plt
#plt.figure(1) # the first figure
#plt.subplot(211) # the first subplot in the first figure
#plt.plot([1, 2, 3])
#plt.subplot(212) # the second subplot in the first figure
#plt.plot([4, 5, 6])
#
#plt.figure(2) # a second figure
#plt.plot([4, 5, 6])
#plt.figure(1) #figure 1 current; subplot(212) still current)
#plt.subplot(211) # make subplot(211) in figure1 current
#plt.title('Easy as 1, 2, 3') #subplot 211 title
#
#plt.show()

#import numpy as np
#import matplotlib.pyplot as plt
#
#mu, sigma = 100, 15
#x = mu + sigma * np.random.randn(10000)
#
## the histogram of the data
#n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)
#
#plt.xlabel('Smarts')
#plt.ylabel('Probability')
#plt.title('Histogram of iq')
#plt.text(60, .025, r'$\mu100, \ \sigma=15$')
#plt.axis([40, 160, 0, 0.03])
#plt.grid(True)
#plt.show()

import numpy as np
import matplotlib.pyplot as plt

ax = plt.subplot(111)
t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)

plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05),)

plt.ylim(-2, 2)
plt.show()
