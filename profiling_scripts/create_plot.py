import numpy as np 
import os
import sys
from datetime import datetime
import matplotlib 
import matplotlib.pyplot as plt

home_directory = "/home/ngraymon"

io_003 	= np.zeros((3,3))
gpu_003 = np.zeros((3,3))
kepler 	= np.zeros((3,3))

#BASIS_SIZE = [100, 200, 300, 350, 400, 450, 500]
basis_label = [27e6, 64e6, 125e6]
BASIS_SIZE = [300, 400, 500]
ITERATIONS = [100, 200, 400]

# first lets gather the cpu (io003) data
for b_index, basis in enumerate(BASIS_SIZE):
	for iter_index, num_iter in enumerate(ITERATIONS):
		filename = "{0}/gpu/profiling_scripts/logs/{3}/{1}_{2}cpu_timing_output.txt".format(home_directory, basis, num_iter, "io003")
		if os.path.isfile(filename):
			try: 
				with open (filename, "r") as input_file:
					for line in input_file:
						if line.startswith("Finshied time:"):
							io_003[b_index][iter_index] = line.split(":")[1]
							#print(io_003[b_index][iter_index])
							#formatted_line = line.split()[1].replace("m", "/").replace(".", "/").replace("s", "").split("/")
							#io_003[b_index][iter_index] = ( int(formatted_line[0]) * 60 ) + int(formatted_line[1]) + (int(formatted_line[2]) / 1000)
							#print("list of words: ", line.split()[1], formatted_line, io_003[b_index][iter_index])

			except IOError as e:
				print("Unable to open ", filename)
	# done
# done
print("Finished reading io003 data")


# now lets gather the gpu003 data
for b_index, basis in enumerate(BASIS_SIZE):
	for iter_index, num_iter in enumerate(ITERATIONS):
		filename = "{0}/gpu/profiling_scripts/logs/{3}/{1}_{2}base64_cpu.txt".format(home_directory, basis, num_iter, "gpu003")
		if os.path.isfile(filename):
			try: 
				with open (filename, "r") as input_file:
					for line in input_file:
						if line.startswith("Time to run"):
							gpu_003[b_index][iter_index] = float(line.split()[3])
			except IOError as e:
				print("Unable to open ", filename)
	# done
# done
print("Finished reading gpu003 data")


# now lets gather the kepler data
for b_index, basis in enumerate(BASIS_SIZE):
	for iter_index, num_iter in enumerate(ITERATIONS):
		filename = "{0}/gpu/profiling_scripts/logs/{3}/{1}_{2}base64_cpu.txt".format(home_directory, basis, num_iter, "kepler")
		if os.path.isfile(filename):
			try: 
				with open (filename, "r") as input_file:
					for line in input_file:
						if line.startswith("Time to run"):
							kepler[b_index][iter_index] = float(line.split()[3])
			except IOError as e:
				print("Unable to open ", filename)
	# done
# done
print("Finished reading kepler data")


# okay now we plot
# setup data structs
datasets = [kepler, gpu_003, io_003]
data_scaled = []
number_of_datasets = len(datasets)
#indicies = np.arange(len(ITERATIONS))
indicies = np.arange(3)
width_one   = 0.15
width_two   = width_one*2
xticklabels = [str(x) for x in ITERATIONS]

# create scaled data
scaled = []
with np.errstate(divide='ignore', invalid='ignore'):
	for indexor, unscaled in enumerate(datasets):
		scaled.append(np.empty_like(unscaled))
		scaled[indexor] = np.divide(io_003, unscaled) 
		scaled[indexor][unscaled == 0] = 0
		scaled[indexor][io_003 == 0] = 0





scaling_on = True if((len(sys.argv) >= 2) and (sys.argv[1] == "scale")) else False 

data_scaled = scaled if(scaling_on) else datasets 

#print(io_003)
#print(scaled_io)
#print(gpu_003)
#print(scaled_gpu)
#print(kepler)
#print(scaled_kepler)
#print(datasets[0], '\n', datasets[1], '\n',  datasets[2], '\n', )
#print(data_scaled[0], '\n',  data_scaled[1], '\n',  data_scaled[2])

font = {'family' : 'monospace',
		'weight' : 'bold',
		'size'   : 32}
matplotlib.rc('font', **font)

def autolabel(rects, data_list):
	string = '{0}x' if(scaling_on) else '{0}s'
	for rect, data_item in zip(rects, data_list):
		height = rect.get_height()
		ax.text(rect.get_x()+rect.get_width()/2., height*(0.99), string.format(int(height)), ha='center', va='top')
		#text_to_print = data_item if(scaling_on) else height
		#ax.annotate(string.format(int(height)), xy = (rect.get_x()+rect.get_width()/2., height), 
		#	xytext= (rect.get_x()+rect.get_width()/2., height-2), fontsize='24')
		
# We dont care about these plots anymore
'''
# compare per basis size
for b_index in range(kepler.shape[0]):
	# compare the things against each other
	figure_others = plt.figure()
	ax = figure_others.add_subplot(111)
	plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
	rects1 = ax.bar(indicies, scaled_io[b_index, :], width_one, color='r')
	rects2 = ax.bar(indicies+width_one, scaled_gpu[b_index, :], width_one, color='y')
	rects3 = ax.bar(indicies+width_two, scaled_kepler[b_index, :], width_one, color='b') #color='k'
	autolabel(rects1)
	autolabel(rects2)
	autolabel(rects3)
	ax.set_ylabel('# times faster')
	ax.set_xlabel('Iterations')
	ax.set_title("Run time for Basis of size {0}".format(BASIS_SIZE[b_index]))
	ax.set_xticks(indicies+width_two)
	ax.set_xticklabels([str(x) for x in ITERATIONS])
	ax.legend((rects1[0], rects2[0], rects3[0]), ('CPU', 'NVIDIA Tesla M2070', 'NVIDIA Geforce GTX Titan Black') )
	figure_others.set_size_inches(22, 22)
	plt.savefig('./plots/{0}_basis_cmp.png'.format(BASIS_SIZE[b_index]), bbox_inches='tight')
	plt.close(figure_others)
print("Finished basis")

'''
indicies = np.arange(len(BASIS_SIZE))
# compare per iteration
for i_index in range(kepler.shape[1]):
	# compare the things against each other
	figure_others = plt.figure()

	ax = figure_others.add_subplot(111)
	plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
	bar = [indicies + width_one, indicies] if(scaling_on) else [indicies + width_two, indicies + width_one, indicies]
	if(scaling_on is False):
		rects1 = ax.bar(bar[2], data_scaled[2][:, i_index], width_one, color='r')
	rects2 = ax.bar(bar[1], data_scaled[1][:, i_index], width_one, color='y')
	rects3 = ax.bar(bar[0], data_scaled[0][:, i_index], width_one, color='c') #color='k'
	if(scaling_on is False):
		autolabel(rects1, datasets[2][:, i_index])
	autolabel(rects2, datasets[1][:, i_index])
	autolabel(rects3, datasets[0][:, i_index])
	ax.set_title("Run time for {0} Iterations".format(ITERATIONS[i_index]), fontsize = 50)
	ax.set_xlabel('Basis size', fontsize = 50)
	ax.set_xticks(bar[0])
	if(scaling_on):
		ax.set_ylabel('Speed up Factor (per CPU time)', fontsize = 50)
		#ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2e'))
		ax.set_xticklabels(['{0:.4G}'.format(x) for x in basis_label], fontsize = 42)
	else:
		ax.set_ylabel('Seconds', fontsize = 50)
		ax.set_xticklabels([str(x) for x in BASIS_SIZE], fontsize = 42)
	if(scaling_on is False):
		ax.legend((rects1[0], rects2[0], rects3[0]), ('Intel Xeon E5-2667', 'NVIDIA Tesla M2070', 'NVIDIA Geforce GTX Titan Black'), bbox_to_anchor=(0.37, 1) )
	else:
		ax.legend((rects2[0], rects3[0]), ('NVIDIA Tesla M2070', 'NVIDIA Geforce GTX Titan Black'), bbox_to_anchor=(1, 1))
	figure_others.set_size_inches(36, 20)
	#plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
	plt.tight_layout()
	plt.savefig('{0}/gpu/profiling_scripts/plots/{1}_iter_cmp.png'.format(home_directory, ITERATIONS[i_index]))
	plt.close(figure_others)
print("Finished iter")

'''
# compare against self
for data in datasets:
	figure_others = plt.figure()
	ax = figure_others.add_subplot(1, 1, b_index+1)
	plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
	ax.set_ylabel('# times faster')
	ax.set_xlabel('Basis size')
	ax.set_title("Run time for Basis of size {0}".format(BASIS_SIZE[b_index]))
	for i_index in range(len(ITERATIONS)):
		rects = ax.bar(indicies, data[:][i_index], width_one, color='r')
		ax.set_xticks(indicies+width_two)
		ax.set_xticklabels(xticklabels)
		ax.legend( (rects1[0], rects2[0], rects3[0]), ('io003', 'gpu003', 'kepler') )
		figure_others.set_size_inches(22, 22)
		plt.savefig('./plots/{0}.png'.format(datasets[data_index]), bbox_inches='tight')
		plt.close(figure_others)

print("Finished self compare")
'''

