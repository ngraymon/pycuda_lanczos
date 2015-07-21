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

# my module
import universe_definitions as ud

# Double precision is only supported by devices with compute capability >= 1.3:
gpu_data_type = np.float32 if misc.get_compute_capability(pycuda.autoinit.device) >= 1.3 else np.float64

# create kinetic component
T_i = np.fromfunction(ud.construct_kinetic_factor, (ud.dimension_n, ud.dimension_n), dtype=np.float64)#.reshape()

# define the three dimensions
x_shape = ()
y_shape = ()
z_shape = ()


#create v 
v_vector = np.asarray(np.random.random_sample(size=(ud.dimension_n, ud.dimension_n, ud.dimension_n)), np.float64)
v_X = v_vector.reshape()
v_Y = v_vector.reshape()
v_Z = v_vector.reshape()
v_X_view = np.ravel(v_X, order='C')
v_Y_view = np.ravel(v_Y, order='C')
v_Z_view = np.ravel(v_Z, order='C')
print(	v_X.shape, v_Y.shape, v_Z.shape, 
		v_X_view.shape, v_Y_view.shape, v_Z_view.shape)


# dont do this!!
# hamiltonian = T_i + T_i + T_i 

# multiply Ti by V 3 times
# need to reshuffle V each time 


#v_y_gpu = gpuarray.to_gpu(v_vector)
#v_x_gpu = gpuarray.to_gpu(v_vector)
#v_z_gpu = gpuarray.to_gpu(v_vector)

# transfer to gpu
T_gpu = gpuarray.to_gpu(T_i)

v_gpu = []
# reshape before hand
v_gpu.append(gpuarray.to_gpu(v_vector))
v_gpu.append(gpuarray.to_gpu(v_vector.reshape((1000000,))))
v_gpu.append(gpuarray.to_gpu(v_vector.reshape((1000000,))))


# reshape afterwards
v_gpu.append(v_gpu[0].copy().reshape((1000000,)))
v_gpu.append(v_gpu[0].reshape((1000000,)).copy())

for vector in v_gpu:
	print("Shape: {0} dtype: {1} mem_size: {2} nbytes: {3} strides: {4} flags: {5} ".format(vector.shape, vector.dtype, vector.mem_size, vector.nbytes, vector.strides, vector.flags))
for vector in v_gpu:
	print(vector)

# the dimensions
# ud.m, ud.n, ud.k, ud.l = 1, 1, 1, 1

# assemble extra parameters
# Dgemm actually preforms (alpha * A) * B + (beta * C) and it expects these parameters
# so we give it beta = 0 and alpha = 1 so that we get the result of (A * B)
alpha 	= np.float64(1.0) # no prefactor
beta 	= np.float64(0.0) # C matrix is not involved so beta = 0.0
c_gpu = gpuarray.empty((ud.l, ud.m, ud.n), np.float64) # an empty matrix of the right size

# these are the 'leading' dimensions of the two-dimensional arrays used to store the matricies
# ud.lda, ud.ldb, ud.ldc = 1, 1, 1

# preform operation
#skcuda.cublas.cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
U_gpu = cublas.cublasDgemm(handle = cublas.cublasCreate(), 
								transa = 'n', 
								transb = 'n',
								m = ud.m, 
								n = ud.n, 
								k = ud.k, 
								alpha = alpha, 
								A = T_gpu.gpudata, 
								lda = ud.lda, 
								B = v_gpu[0].gpudata, 
								ldb = ud.ldb, 
								beta = beta, 
								C = c_gpu.gpudata, 
								ldc = ud.ldc)

# reshuffle to allow the T_x and T_y multiplication as Dgemm() call
# v2_gpu = shuffl(v_gpu)

# print out to check for success
print('Result of our douple precision dot product\n', U_gpu.get()) 










