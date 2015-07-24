# profiling
import time

# running on different machines
import platform

# pycuda stuff
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
import numpy as np

#scikit stuff
import scipy.constants as s_const
import skcuda.linalg as linalg
import skcuda.cublas as cublas
import skcuda.misc as misc
import string
linalg.init()


# my module
import universe_definitions as ud

#BASIS_SIZE = [100, 200, 300, 350, 400, 450, 500] 
#ITERATIONS = [100, 200, 400]


BASIS_SIZE = [100, 200, 300, 400, 500] 
ITERATIONS = [100, 200, 400]

x_shape = (0, 1, 2)
y_shape = (1, 0, 2)
z_shape = (2, 1, 0)

v_x_gpu = None
v_y_gpu = None
v_z_gpu = None
T_gpu   = None
U_x_gpu = None
U_y_gpu = None
U_z_gpu = None

def prepare_gpu(basis_size):
	if misc.get_compute_capability(pycuda.autoinit.device) >= 1.3:
		gpu_data_type = np.float64
		print("Data type", str(gpu_data_type))
	else:
		print("NO DOUBLE PRECIISION OMG!")
		exit(0)

	T_i = np.fromfunction(ud.construct_kinetic_factor, (basis_size, basis_size), dtype=np.float64)

	# assume that we are in i,j,k arrangement initially
	v_vector = np.asarray(np.random.random_sample(size=(basis_size, basis_size, basis_size)), dtype=np.float64, order='C')

	# arrange the vector data on the gpu
	v_x_gpu = gpuarray.to_gpu(v_vector)
	#v_y_gpu = gpuarray.to_gpu(v_vector.transpose(y_shape).copy(order='C'))
	#v_z_gpu = gpuarray.to_gpu(v_vector.transpose(y_shape).copy(order='C'))

	# transfer kinetic portion to gpu
	T_gpu   = gpuarray.to_gpu(T_i)

	# allocate space on gpu for results
	U_x_gpu = gpuarray.zeros((basis_size, basis_size*basis_size), np.float64) # an empty matrix of the right size
	#U_y_gpu = gpuarray.zeros((basis_size, basis_size*basis_size), np.float64) # an empty matrix of the right size
	#U_z_gpu = gpuarray.zeros((basis_size, basis_size*basis_size), np.float64) # an empty matrix of the right size
	#return T_gpu, v_x_gpu, v_y_gpu, v_z_gpu, U_x_gpu, U_y_gpu, U_z_gpu
	return T_gpu, v_x_gpu, U_x_gpu,

# create two timers so we can speed-test each approach
start = drv.Event()
end = drv.Event()


machine = "kepler" if(platform.release().startswith("2.6.32")) else "gpu003"
for basis in BASIS_SIZE:
	print("Basis size:", basis)
	T_gpu, v_x_gpu, U_x_gpu, = prepare_gpu(basis) # set it up
	#T_gpu, v_x_gpu, v_y_gpu, v_z_gpu, U_x_gpu, U_y_gpu, U_z_gpu = prepare_gpu(basis) # set it up
	i, j_k, i_prime = basis, basis*basis, basis
	print("About to start iterations")
	for num_iter in ITERATIONS:
		print("ITERATION:", num_iter)
		start.record()
		for place_holder in range(num_iter):
			cublas.cublasDgemm(handle = cublas.cublasCreate(), 
								transa = 'n', transb = 'n',
								m 	= i, n 	= j_k, 		k = i_prime,
								lda = i, ldb = i_prime, ldc = i,
								alpha = ud.alpha,  beta = ud.beta, 
								A = T_gpu.gpudata, 
								B = v_x_gpu.gpudata, 
								C = U_x_gpu.gpudata, )
			cublas.cublasDgemm(handle = cublas.cublasCreate(), 
								transa = 'n', transb = 'n',
								m 	= i, n 	= j_k, 		k = i_prime,
								lda = i, ldb = i_prime, ldc = i,
								alpha = ud.alpha,  beta = ud.beta, 
								A = T_gpu.gpudata, 
								B = v_x_gpu.gpudata, 
								C = U_x_gpu.gpudata, )
			cublas.cublasDgemm(handle = cublas.cublasCreate(), 
								transa = 'n', transb = 'n',
								m 	= i, n 	= j_k, 		k = i_prime,
								lda = i, ldb = i_prime, ldc = i,
								alpha = ud.alpha,  beta = ud.beta, 
								A = T_gpu.gpudata, 
								B = v_x_gpu.gpudata, 
								C = U_x_gpu.gpudata, )
			'''
			cublas.cublasDgemm(handle = cublas.cublasCreate(), 
								transa = 'n', transb = 'n',
								m 	= i, n 	= j_k, 		k = i_prime,
								lda = i, ldb = i_prime, ldc = i,
								alpha = ud.alpha,  beta = ud.beta, 
								A = T_gpu.gpudata, 
								B = v_y_gpu.gpudata, 
								C = U_y_gpu.gpudata, )
			cublas.cublasDgemm(handle = cublas.cublasCreate(), 
								transa = 'n', transb = 'n',
								m 	= i, n 	= j_k, 		k = i_prime,
								lda = i, ldb = i_prime, ldc = i,
								alpha = ud.alpha,  beta = ud.beta, 
								A = T_gpu.gpudata, 
								B = v_z_gpu.gpudata, 
								C = U_z_gpu.gpudata, )
			'''
			# we are not going to do the potential for now
		end.record()
		end.synchronize()
		finish_time = start.time_till(end)*1e-3
		#print("Python time:", finish_time, "\n vs \n GPU time: ", secs)
		with open ("./logs/{2}/{0}_{1}base64_cpu.txt".format(basis, num_iter, machine), "w+") as output_file:
			print("Basis size: ", basis, 
				"\nNumber of iterations: ", num_iter,
				"\nTime to run ", finish_time,
				file = output_file)
		print("Finished writing to file")
	time.sleep(3)
	# done
# done
	



