import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
import numpy as np

import scipy.constants as s_const
import skcuda.linalg as linalg
import skcuda.cublas as cublas
import skcuda.misc as misc
import string
linalg.init()

# Double precision is only supported by devices with compute capability >= 1.3:
gpu_data_type = np.float32 if misc.get_compute_capability(pycuda.autoinit.device) >= 1.3 else np.float64

# define some constants
dimension_n = 100
m = 1
n = 1
k = 1
l = 1
hbar 	= 1
mass 	= 1
delta_x = (2 * l / dimension_n)

# this is the fraction in front
fractional_constant = ( (hbar **2) / (2 * mass * delta_x) )

# define function, this is the step_wise portion
def step_wise(i, i_prime):
	if(i == i_prime):
		return ( (s_const.pi**2) / 3 )
	else:
		return ( 2 / ((i - i_prime)**2) )

# this is the formula for constructing a single term T_i_iprime
def construct_single_term_in_kinetic_factor(i, i_prime):
	[fractional_constant * ((-1)**(i-i_prime)) * step_wise(i, i_prime) for (i, i_prime) in  (range(dimension_n), range(dimension_n))]

# operate on vectors of i, i_prime
construct_kinetic_factor = np.vectorize(construct_single_term_in_kinetic_factor)

# create kinetic component
T_i = np.fromfunction(construct_kinetic_factor, (dimension_n, dimension_n), dtype=np.float64)#.reshape()

#create v 
v_vector = np.asarray(np.random.rand(dimension_n**3), np.float64)

# dont do this!!
# hamiltonian = T_i + T_i + T_i 

# multiply Ti by V 3 times
# need to reshuffle V each time 


#v_y_gpu = gpuarray.to_gpu(v_vector)
#v_x_gpu = gpuarray.to_gpu(v_vector)
#v_z_gpu = gpuarray.to_gpu(v_vector)

# transfer to gpu
T_gpu = gpuarray.to_gpu(T_i)
v_gpu = gpuarray.to_gpu(v_vector)

# assemble extra parameters
# Dgemm actually preforms (alpha * A) * B + (beta * C) and it expects these parameters
# so we give it beta = 0 and alpha = 1 so that we get the result of (A * B)
alpha 	= np.float64(1.0) # no prefactor
beta 	= np.float64(0.0) # C matrix is not involved so beta = 0.0
c_gpu = gpuarray.empty((l, m, n), np.float64) # an empty matrix of the right size

# these are the 'leading' dimensions of the two-dimensional arrays used to store the matricies
lda, ldb, ldc = 1, 1, 1
# preform operation
#skcuda.cublas.cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
U_gpu = cublas.cublasDgemm(handle = cublas.cublasCreate(), 
								transa = 'n', 
								transb = 'n',
								m = m, 
								n = n, 
								k = k, 
								alpha = alpha, 
								A = T_gpu.gpudata, 
								lda = lda, 
								B = v_gpu.gpudata, 
								ldb = ldb, 
								beta = beta, 
								C = c_gpu.gpudata, 
								ldc = ldc)

# reshuffle to allow the T_x and T_y multiplication as Dgemm() call
# v2_gpu = shuffl(v_gpu)

# print out to check for success
print('Result of our douple precision dot product\n', U_gpu.get()) 










