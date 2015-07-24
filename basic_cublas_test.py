# profiling
import time

def timefunc(f):
	def f_timer(*args, **kwargs):
		start = time.time()
		result = f(*args, **kwargs)
		end = time.time()
		print(f.__name__, 'took', end - start, 'time')
		return result
	return f_timer


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

# Double precision is only supported by devices with compute capability >= 1.3:
gpu_data_type = np.float64 if misc.get_compute_capability(pycuda.autoinit.device) >= 1.3 else np.float32

# create kinetic component
T_i = np.fromfunction(ud.construct_kinetic_factor, (ud.basis_size, ud.basis_size), dtype=np.float64)
print("Finished creating our square kinetic matrix\n")


t0 = time.clock()
#potential = np.fromfunction(ud.construct_potential_factor, (ud.basis_size, ud.basis_size, ud.basis_size), dtype=np.float64)
print(time.clock() - t0, "seconds process time")
#print("Kinetic Term: ", T_i.size, "Potential size: ", potential.size)

# define the three dimensions
x_shape = (0, 1, 2)
y_shape = (1, 0, 2)
z_shape = (2, 1, 0)

 
# assume that we are in i,j,k arrangement initially
v_vector = np.asarray(np.random.random_sample(size=(ud.basis_size, ud.basis_size, ud.basis_size)), dtype=np.float64, order='C')

v_X = v_vector #v_vector.transpose(x_shape).copy(order='C')					# i, j, k
# swap i, j
v_Y = v_vector.transpose(y_shape).copy(order='C')	# j, i, k
# swap i, k
v_Z = v_vector.transpose(z_shape).copy(order='C')
print("Finished creating our three vectors\n")

'''
print(	#"Original vector flattened in C order", v_vector.flatten(order='C'),
		'\n', "Do the three vectors have the same shape?: ", (v_X.shape == v_Y.shape == v_Z.shape), 
		'\n', "Are vectors v_X and v_X equal?: ", np.array_equal(v_X, v_X),
		'\n', "Are vectors v_Y and v_Y equal?: ", np.array_equal(v_Y, v_Y),
		'\n', "Are vectors v_Z and v_Z equal?: ", np.array_equal(v_Z, v_Z),
		'\n', "Are vectors v_X and v_Y equal?: ", np.array_equal(v_X, v_Y),
		'\n', "Are vectors v_X and v_Z equal?: ", np.array_equal(v_X, v_Z),
		'\n', "Are vectors v_Y and v_Z equal?: ", np.array_equal(v_Y, v_Z)
	)
assert(v_X.shape == v_Y.shape == v_Z.shape)
'''
# preshaped vector
v_x_gpu = gpuarray.to_gpu(v_X)
v_y_gpu = gpuarray.to_gpu(v_Y)
v_z_gpu = gpuarray.to_gpu(v_Z)

'''
print(	"Do the three gpu vectors have the same shape?: ", (v_x_gpu.shape == v_y_gpu.shape == v_z_gpu.shape), 
		'\n', "Are vectors v_x_gpu and v_x_gpu equal?: ", np.array_equal(v_x_gpu.get(), v_x_gpu.get()),
		'\n', "Are vectors v_y_gpu and v_y_gpu equal?: ", np.array_equal(v_y_gpu.get(), v_y_gpu.get()),
		'\n', "Are vectors v_z_gpu and v_z_gpu equal?: ", np.array_equal(v_z_gpu.get(), v_z_gpu.get()),
		'\n', "Are vectors v_x_gpu and v_y_gpu equal?: ", np.array_equal(v_x_gpu.get(), v_y_gpu.get()),
		'\n', "Are vectors v_x_gpu and v_z_gpu equal?: ", np.array_equal(v_x_gpu.get(), v_z_gpu.get()),
		'\n', "Are vectors v_y_gpu and v_z_gpu equal?: ", np.array_equal(v_y_gpu.get(), v_z_gpu.get()), 
		'\n',
		'\n', "Are vectors v_x_gpu and v_X equal?: ", np.array_equal(v_x_gpu.get(), v_X),
		'\n', "Are vectors v_y_gpu and v_Y equal?: ", np.array_equal(v_y_gpu.get(), v_Y),
		'\n', "Are vectors v_z_gpu and v_Z equal?: ", np.array_equal(v_z_gpu.get(), v_Z),
		'\n', "Are vectors v_x_gpu and v_Y equal?: ", np.array_equal(v_x_gpu.get(), v_Y),
		'\n', "Are vectors v_x_gpu and v_Z equal?: ", np.array_equal(v_x_gpu.get(), v_Z),
		'\n', "Are vectors v_y_gpu and v_Z equal?: ", np.array_equal(v_y_gpu.get(), v_Z),
		'\n',
	)
'''

#print("X", v_x_gpu.get() == v_X, "Y", v_y_gpu.get() == v_Y, "Z", v_z_gpu.get() == v_Z)

#print("\nx\n", v_x_gpu, "\ny\n", v_y_gpu, "\nz\n", v_z_gpu)

# transfer to gpu
T_gpu = gpuarray.to_gpu(T_i)
#pot_gpu = gpuarray.to_gpu(potential.reshape( (ud.basis_size, ud.basis_size*ud.basis_size) ).copy())
#print("Potential is properly shaped? ", (pot_gpu.get().shape == potential.reshape( (ud.basis_size, ud.basis_size*ud.basis_size) ).shape))

'''
v_gpu = []
 reshape afterwards
v_gpu.append(v_gpu[0].copy().reshape((1000000,)))
v_gpu.append(v_gpu[0].reshape((1000000,)).copy())


for vector in v_gpu:
	print("Shape: {0} dtype: {1} mem_size: {2} nbytes: {3} strides: {4} flags: {5} ".format(vector.shape, vector.dtype, vector.mem_size, vector.nbytes, vector.strides, vector.flags))
for vector in v_gpu:
	print(vector)
'''

# the dimensions
# ud.m, ud.n, ud.k, ud.l = 1, 1, 1, 1

print("Okay now we are going to allocate space on the cpu and gpu\n")

# allocate space on gpu
U_x_gpu 		= gpuarray.zeros((ud.basis_size, ud.basis_size*ud.basis_size), np.float64) # an empty matrix of the right size
U_y_gpu 		= gpuarray.zeros((ud.basis_size, ud.basis_size*ud.basis_size), np.float64) # an empty matrix of the right size
U_z_gpu 		= gpuarray.zeros((ud.basis_size, ud.basis_size*ud.basis_size), np.float64) # an empty matrix of the right size
potential_gpu 	= gpuarray.zeros((ud.basis_size, ud.basis_size, ud.basis_size), np.float64) # an empty matrix of the right size
hamiltonian_gpu = gpuarray.zeros((ud.basis_size, ud.basis_size, ud.basis_size), np.float64) # an empty matrix of the right size

# allocate space for cpu
u_x_cpu = np.zeros_like(v_X, dtype = np.float64, order = 'C')
u_y_cpu = np.zeros_like(v_Y, dtype = np.float64, order = 'C')
u_z_cpu = np.zeros_like(v_Z, dtype = np.float64, order = 'C')
#pot_cpu = np.zeros_like(potential, dtype = np.float64, order = 'C')
#hamiltonian_cpu = np.zeros_like(potential, dtype = np.float64, order = 'C')
#hamiltonian_check = np.zeros_like(potential, dtype = np.float64, order = 'C')

print("Okay we finished allocating space on the cpu and the gpu, time to do work\n")

# preform operation
#skcuda.cublas.cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
#def Hv(v_matrix, T_i, ):
@timefunc
def mult_BLAS():
	alpha 	= np.float64(1.0) # no prefactor
	beta 	= np.float64(0.0) # C matrix is not involved so beta = 0.0
	#m, k, n = ud.basis_size, ud.basis_size, ud.basis_size**2
	t0 = time.clock()
	for a in range(100):
		cublas.cublasDgemm(handle = cublas.cublasCreate(), 
							transa = 'n', transb = 'n',
							m 	= ud.i, n 	= ud.j_k, 		k = ud.i_prime,
							lda = ud.i, ldb = ud.i_prime, ldc = ud.i,
							alpha = alpha,  beta = beta, 
							A = T_gpu.gpudata, 
							B = v_x_gpu.gpudata, 
							C = U_x_gpu.gpudata, )
		cublas.cublasDgemm(handle = cublas.cublasCreate(), 
							transa = 'n', transb = 'n',
							m 	= ud.i, n 	= ud.j_k, 		k = ud.i_prime,
							lda = ud.i, ldb = ud.i_prime, ldc = ud.i,
							alpha = alpha,  beta = beta, 
							A = T_gpu.gpudata, 
							B = v_y_gpu.gpudata, 
							C = U_y_gpu.gpudata, )
		cublas.cublasDgemm(handle = cublas.cublasCreate(), 
							transa = 'n', transb = 'n',
							m 	= ud.i, n 	= ud.j_k, 		k = ud.i_prime,
							lda = ud.i, ldb = ud.i_prime, ldc = ud.i_prime,
							alpha = alpha,  beta = beta, 
							A = T_gpu.gpudata, 
							B = v_z_gpu.gpudata, 
							C = U_z_gpu.gpudata, )
		'''cublas.cublasDgemm(handle = cublas.cublasCreate(), 
							transa = 'n', transb = 'n',
							m 	= ud.i, n 	= ud.j_k, 		k = ud.i_prime,
							lda = ud.i, ldb = ud.i_prime, ldc = ud.i,
							alpha = alpha,  beta = beta, 
							A = pot_gpu.gpudata, 
							B = v_x_gpu.gpudata, 
							C = potential_gpu.gpudata, )'''
	print(time.clock() - t0, "mult_BLAS timer")
	return

# this should be a NON blas implementation of the Dgemm on the GPU
@timefunc
def mult_pycuda():
	pass

'''
@timefunc
def mult_cpu():

	#print(	"\nx shape", u_x_cpu.shape, 
	#		"\ny shape", u_y_cpu.shape, 
	#		"\nz shape", u_z_cpu.shape, 
	#		)

	print("T_i shape:", T_i.shape, "v_X_shape: ", v_X.shape)
	u_x_cpu = np.dot(T_i, v_X.reshape((ud.basis_size, ud.basis_size*ud.basis_size)))
	sample = [ u_x_cpu.flatten()[x] for x in range(10) ]
	print("dot Shape: ", u_x_cpu.shape, "sampling", sample)

	a = np.einsum("pjk,ip -> ijk", v_X, T_i)
	#sample = [ a.flatten()[x] for x in range(10) ]
	#with open ("cpu.txt", "w") as cpu_file:
	#	print("einsum Shape: ", a.shape, "\nsampling\n", a, file = cpu_file)
	
	#sample = [ U_x_gpu.get().flatten()[x] for x in range(10) ]

	with open ("gpu.txt", "w") as gpu_file:
		print("GPU Shape: ", U_x_gpu.get().shape, "\nsampling\n",  np.array_str(U_x_gpu.get(), max_line_width = 75, precision= 5, suppress_small = True), file = gpu_file)



	u_x_cpu = np.tensordot(T_i, v_X, ([0], [0]))
	sample = [ u_x_cpu.flatten()[x] for x in range(10) ]
	print("00 Shape: ", u_x_cpu.shape, "sampling",  sample)

	u_x_cpu = np.tensordot(T_i, v_X, ([0], [1]))
	sample = [ u_x_cpu.flatten()[x] for x in range(10) ]
	print("01 Shape: ", u_x_cpu.shape, "sampling",  sample)

	u_x_cpu = np.tensordot(T_i, v_X, ([1], [0]))
	sample = [ u_x_cpu.flatten()[x] for x in range(10) ]
	print("10 Shape: ", u_x_cpu.shape, "sampling",  sample)

	u_x_cpu = np.tensordot(T_i, v_X, ([1], [1]))
	sample = [ u_x_cpu.flatten()[x] for x in range(10) ]
	print("11 Shape: ", u_x_cpu.shape, "sampling",  sample)

	u_x_cpu = np.tensordot(T_i, v_X, ([0], [2]))
	sample = [ u_x_cpu.flatten()[x] for x in range(10) ]
	print("02 Shape: ", u_x_cpu.shape, "sampling",  sample)
	with open ("base_cpu.txt", "w") as cpu_file:
		print("einsum Shape: ", u_x_cpu.shape, "\nsampling\n",  np.array_str(u_x_cpu, max_line_width = 75, precision= 5, suppress_small = True), file = cpu_file)
	with open ("reshaped_cpu.txt", "w") as cpu_file:
		print("einsum Shape: ", np.swapaxes(u_x_cpu,0,2).reshape((ud.basis_size, ud.basis_size*ud.basis_size)).shape, "\nsampling\n",  np.array_str(np.swapaxes(u_x_cpu,0,2).reshape((ud.basis_size, ud.basis_size*ud.basis_size)), max_line_width = 75, precision= 5, suppress_small = True), file = cpu_file)

	u_y_cpu = np.dot(T_i, v_Y.reshape((ud.basis_size, ud.basis_size*ud.basis_size)))
	u_z_cpu = np.dot(T_i, v_Z.reshape((ud.basis_size, ud.basis_size*ud.basis_size)))
	return

@timefunc
def check_kinetic_parts():
	print("\nDo we have the same T?", np.array_equal(T_i, T_gpu.get()), 
		  )
	print("\nDo we have the same vector x?", np.array_equal(v_X, v_x_gpu.get()), 
		  "\nDo we have the same vector y?", np.array_equal(v_Y, v_y_gpu.get()),  
		  "\nDo we have the same vector z?", np.array_equal(v_Z, v_z_gpu.get()),
		  )	
	print("\nDid we get the correct kinetic x portion?", np.array_equal(U_x_gpu.get(), u_x_cpu.reshape((ud.basis_size, ud.basis_size*ud.basis_size))), 
		  "\nDid we get the correct kinetic y portion?", np.array_equal(U_y_gpu.get(), u_y_cpu.reshape((ud.basis_size, ud.basis_size*ud.basis_size))),  
		  "\nDid we get the correct kinetic z portion?", np.array_equal(U_z_gpu.get(), u_z_cpu.reshape((ud.basis_size, ud.basis_size*ud.basis_size))),
		  )
	return

@timefunc
def add_gpu():
	hamiltonian_gpu = U_x_gpu + U_y_gpu + U_z_gpu + potential_gpu
	#hamiltonian_cpu = U_x_gpu.get() + U_y_gpu.get() + U_z_gpu.get() + potential_gpu.get()
	#print("GPU?", hamiltonian_gpu)
	return

@timefunc
def add_cpu():  
	hamiltonian_check = u_x_cpu + u_y_cpu + u_z_cpu + pot_cpu # dont have to reshape
	#print("CPU?", hamiltonian_check)
	return
'''
# do the kinetic part
mult_BLAS()
exit(0)
mult_cpu()
print(	
		"\nu_x_cpu shape",	u_x_cpu.reshape((ud.basis_size, ud.basis_size*ud.basis_size)).shape,
		"\nu_x_gpu shape",	U_x_gpu.get().shape,
		"\nu_x_cpu", 		[ u_x_cpu.flatten()[x] for x in range(10) ],
		"\nu_x_gpu", 		[ U_x_gpu.get().flatten()[x] for x in range(10) ],
		)
check_kinetic_parts()

# do the potential part
potential_gpu = pot_gpu   * v_x_gpu
pot_cpu 	  = potential * v_X

# add it up
add_gpu()
add_cpu()

# retrive it from the gpu
hamiltonian_cpu = hamiltonian_gpu.get()

#print("GPU?", hamiltonian_cpu)
#print("CHECK?", hamiltonian_check)

U_x_cpu = U_x_gpu.get()
U_y_cpu = U_y_gpu.get()
U_z_cpu = U_z_gpu.get()
potential_cpu = potential_gpu.get()

# reshuffle to allow the T_x and T_y multiplication as Dgemm() call
# v2_gpu = shuffl(v_gpu)

# print out to check for success
print('Result of our douple precision dot product', 
	"\nX shape: ", U_x_cpu.shape,
	"\nY shape: ", U_y_cpu.shape,
	"\nZ shape: ", U_z_cpu.shape,
	'\n', "Do the three vectors have the same shape?: ", (U_x_cpu.shape == U_y_cpu.shape == U_z_cpu.shape), 
	'\n', "Are vectors v_X and v_Y equal?: ", np.array_equal(U_x_cpu, U_y_cpu),
	'\n', "Are vectors v_X and v_Z equal?: ", np.array_equal(U_x_cpu, U_z_cpu),
	'\n', "Are vectors v_Y and v_Z equal?: ", np.array_equal(U_y_cpu, U_z_cpu),
	"\n potential_cpu shape: ", potential_cpu.shape,
	"\n hamiltonian_cpu shape: ", hamiltonian_cpu.shape, 
	"\n potential_gpu shape: ", potential_gpu.shape,
	"\n hamiltonian_gpu shape: ", hamiltonian_gpu.get().shape, 
	"\n Does our hamiltonian match up?", np.array_equal(hamiltonian_cpu, hamiltonian_gpu.get()),
	) 


exit(0)

'''
for iteration in ud.number_of_iterations:
	U_gpu = Hv(v0,Tx,size,size3d,pot); 


	if(iteration%100 is 0):
		print("Iteration ", iteration)

for (j=1;j<=niter;j++) {
	
	r=r+u;
	
	alpha(j-1)=v0*r;
	r=r-(alpha(j-1)*v0);
	
	beta2(j)=r*r;
	beta(j)=sqrt(beta2(j));
	r=(1./beta(j))*r; // to get v
	v0=(-beta(j))*v0; // prepare r check minus sign!!!

	u=v0;     // swapping
	v0=r;
	r=u;

}  
'''









