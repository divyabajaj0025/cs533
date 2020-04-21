import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('pgf')
rc_xelatex = {'pgf.rcfonts': True}
matplotlib.rcParams.update(rc_xelatex)

map_size = [8, 16, 32]
sync_v1 = [0.8688204288482666, 19.028144359588623, 471.98448061943054]
sync_v2 = [0.29715871810913086, 1.8688197135925293, 13.882546663284302]
dist_v1 = [6.340137481689453, 60.673938274383545, 1261.5731217861176]
dist_v2 = [1.306192398071289, 1.7891209125518799, 10.558962106704712]
plt.plot(map_size, sync_v1)
plt.plot(map_size, sync_v2)
plt.plot(map_size, dist_v1)
plt.plot(map_size, dist_v2)
plt.xlabel("MAP Size")
plt.ylabel("Execution Time(second)")
plt.legend(labels=["Sync V1","Sync V2", "Dist V1", "Dist V2"], loc = 'best', ncol=2, fontsize='x-small')
plt.savefig("map_size.pgf")
plt.close()


#map_size = [8, 16, 32]
#sync_v1 = [0.8688204288482666, 19.028144359588623, 471.98448061943054]
#sync_v2 = [0.29715871810913086, 1.8688197135925293, 13.882546663284302]
#dist_v1 = [6.340137481689453, 60.673938274383545, 1261.5731217861176]
#dist_v2 = [1.306192398071289, 1.7891209125518799, 10.558962106704712]
#plt.plot(map_size, sync_v1)
plt.plot(map_size, sync_v2)
#plt.plot(map_size, dist_v1)
plt.plot(map_size, dist_v2)
plt.xlabel("MAP Size")
plt.ylabel("Execution Time(second)")
plt.legend(labels=["Sync V2", "Dist V2"], loc = 'best', ncol=2, fontsize='x-small')
plt.savefig("v2.pgf")
plt.close()

workers_num = [2, 4, 8]
dist_v11 = [9.388800859451294, 6.2014594078063965, 5.429482460021973]
dist_v22 = [0.7465131282806396, 2.2660014629364014, 0.8415665626525879]
plt.plot(workers_num, dist_v11)
plt.plot(workers_num, dist_v22)
plt.xlabel("Number of workers")
plt.ylabel("Execution Time(second)")
plt.legend(labels=["Dist V1","Dist V2"], loc = 'best', ncol=2, fontsize='x-small')
plt.savefig("workers.pgf")
plt.close()
