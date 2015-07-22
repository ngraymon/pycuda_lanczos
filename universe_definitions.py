import numpy as np

# define some constants
w = [1, 2, 3] #i,j,k
w2 = np.multiply(w, w)
print("w2", w2)
basis_size = 400
print("basis_size", basis_size)
number_of_iterations = 100
V_ceiling = 100
grid_start = -10
grid_finish = 10
hbar, mass 	= 1, 1
delta_x = ( (grid_finish - grid_start) / basis_size )

# define the dimensions
i = basis_size
i_prime = basis_size
j_k = basis_size*basis_size

# these are the 'leading' dimensions of the two-dimensional arrays used to store the matricies
lda, ldb, ldc = i, i_prime, j_k

# this is the fraction in front
fractional_constant = ( (hbar**2) / (2 * mass * delta_x) )

# define function, this is the step_wise portion
def step_wise(i, i_prime):
	return ( (np.pi**2) / 3 ) if (i == i_prime) else ( 2 / ((i - i_prime)**2) )

# this is the formula for constructing a single term T_i_iprime
def construct_single_term_in_kinetic_factor(i, i_prime):
	return fractional_constant * ((-1)**(i-i_prime)) * step_wise(i, i_prime)

# operate on vectors of i, i_prime
construct_kinetic_factor = np.vectorize(construct_single_term_in_kinetic_factor)

'''
def omega_dimension(w2_array, dimension_array):
	assert(w2_array.size == dimension_array.size) # make sure they are the same size
	return_val = 0
	for w2_item, dimension_item in zip(w2_array, dimension_array):
		return_val += w2_item * ((grid_start + dimension_item*delta_x)**2)
	return return_val

def construct_single_term_in_potential_factor(i, j, k):
	return 0.5 * mass * omega_dimension(w2, np.array([i, j, k]))

'''
def construct_single_term_in_potential_factor(i, j, k):
	return 0
	#0.5 * mass * np.sum(w2 * ((grid_start + np.array([i, j, k])*delta_x)**2))

# operate on vectors of i, i_prime
construct_potential_factor = np.vectorize(construct_single_term_in_potential_factor)



#Input:
## int size 
## int 		size
## int 		niter
## double 	Vceil
## double 	Ri
## double 	Rf
##
##
##
##