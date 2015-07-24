import numpy as np 
import os
from datetime import datetime
import matplotlib 
import matplotlib.pyplot as plt

io_003 	= np.zeros((5,3))
gpu_003 = np.zeros((5,3))
kepler 	= np.zeros((5,3))

#BASIS_SIZE = [100, 200, 300, 350, 400, 450, 500]
BASIS_SIZE = [100, 200, 300, 400, 500]
ITERATIONS = [100, 200, 400]

# first lets gather the cpu (io003) data
for b_index, basis in enumerate(BASIS_SIZE):
	for iter_index, num_iter in enumerate(ITERATIONS):
		filename = "./logs/{2}/{0}_{1}cpu_timing_output.txt".format(basis, num_iter, "io003")
		if os.path.isfile(filename):
			try: 
				with open (filename, "r") as input_file:
					for line in input_file:
						if line.startswith("Finshied time:"):
							io_003[b_index][iter_index] = line.split(":")[1]
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
		filename = "./logs/{2}/{0}_{1}cpu_timing_output.txt".format(basis, num_iter, "io003")
		if os.path.isfile(filename):
			try: 
				with open ("./logs/{2}/{0}_{1}base64_cpu.txt".format(basis, num_iter, "gpu003"), "r") as input_file:
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
		filename = "./logs/{2}/{0}_{1}cpu_timing_output.txt".format(basis, num_iter, "io003")
		if os.path.isfile(filename):
			try: 
				with open ("./logs/{2}/{0}_{1}base64_cpu.txt".format(basis, num_iter, "kepler"), "r") as input_file:
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
width_one   = 0.35
width_two   = width_one*2
xticklabels = [str(x) for x in ITERATIONS]

# create scaled data
#print(io_003, gpu_003, kepler)
print("check2")
scaled = []
with np.errstate(divide='ignore'):
	for indexor, unscaled in enumerate(datasets):
		scaled.append(np.empty_like(unscaled))
		scaled[indexor] = np.divide(io_003, unscaled) 
		scaled[indexor][unscaled == 0] = 0
		scaled[indexor][io_003 == 0] = 0

scaling_on = True

data_scaled = scaled if(scaling_on) else datasets 

#print(io_003)
#print(scaled_io)
#print(gpu_003)
#print(scaled_gpu)
#print(kepler)
#print(scaled_kepler)
#print(datasets[0], '\n', datasets[1], '\n',  datasets[2], '\n', )
#print(scaled_io, '\n',  scaled_gpu, '\n',  scaled_kepler)

font = {'family' : 'monospace',
		'weight' : 'bold',
		'size'   : 24}
matplotlib.rc('font', **font)

def autolabel(rects):
	
	for rect in rects:
		height = rect.get_height()
		ax.text(rect.get_x()+rect.get_width()/2., 0.41+height, '%d'%int(height),
				ha='center', va='bottom')

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
	rects1 = ax.bar(indicies, data_scaled[2][:, i_index], width_one, color='r')
	rects2 = ax.bar(indicies+width_one, data_scaled[1][:, i_index], width_one, color='y')
	rects3 = ax.bar(indicies+width_two, data_scaled[0][:, i_index], width_one, color='b') #color='k'
	autolabel(rects1)
	autolabel(rects2)
	autolabel(rects3)
	if(scaling_on):
		ax.set_ylabel('Speed up Factor (per CPU time)')
	else:
		ax.set_ylabel('Seconds')

	ax.set_xlabel('Basis size')
	ax.set_title("Run time for {0} Iterations".format(ITERATIONS[i_index]))
	ax.set_xticks(indicies+width_two)
	ax.set_xticklabels([str(x) for x in BASIS_SIZE])
	ax.legend((rects1[0], rects2[0], rects3[0]), ('CPU', 'NVIDIA Tesla M2070', 'NVIDIA Geforce GTX Titan Black') )
	figure_others.set_size_inches(22, 22)
	plt.savefig('./plots/{0}_iter_cmp.png'.format(ITERATIONS[i_index]), bbox_inches='tight')
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

