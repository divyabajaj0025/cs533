import matplotlib
from matplotlib import pyplot as plt
#matplotlib.use('pgf')
#rc_xelatex = {'pgf.rcfonts': True}
#matplotlib.rcParams.update(rc_xelatex)

cpus = [1, 2, 4, 8]
total_time = [6.135682582855225, 3.219874858856201, 1.9219090938568115, 1.921813726425171]
plt.plot(cpus, total_time)
plt.xlabel("# CPUs")
plt.ylabel("total_time")
for i_x, i_y in zip(cpus, total_time):
    plt.text(i_x - 1.25, i_y + 0.25, '({}, {})'.format(i_x, i_y))
plt.savefig("map_reduce.png")
plt.close()
