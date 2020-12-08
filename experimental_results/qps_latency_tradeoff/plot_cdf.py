import matplotlib.pyplot as plt
import numpy as np
import sys

elapsed_latency_list = []
path = sys.argv[1]
QPS = sys.argv[2]
latency_SLO = sys.argv[3]
partitions = sys.argv[4]
filename = path + "/latencies.txt"
with open(filename, 'r') as reader:
    # Read and print the entire file line by line
    line = reader.readline()
    while line != '':  # The EOF char is an empty string
        #print(line, end='')
        elapsed_latency_list.append(float(line))
        line = reader.readline()
#print(elapsed_latency_list)

p = np.linspace(0, 1, len(elapsed_latency_list), endpoint=False)

custom_label = "Latency CDF for QPS: " + str(QPS) + ", Latency SLO: " + str(latency_SLO) + ", # partitions: " + str(partitions)
# plot the sorted data:
plt.figure()

plt.title(label=custom_label, 
          fontsize=10, 
          color="red")

plt.plot(elapsed_latency_list,p)
plt.ylabel('Latency Percentile')
plt.xlabel('Latency Range')
png_name = "latency_cdf_qps_ " + str(QPS) + "l_" + str(latency_SLO) + "p_" + str(partitions) + ".png"
plt.savefig(path + "/"+ png_name)
