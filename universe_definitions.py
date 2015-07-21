import numpy as np

# define the dimensions
m, n, k, l = 1, 1, 1, 1

# these are the 'leading' dimensions of the two-dimensional arrays used to store the matricies
lda, ldb, ldc = 1, 1, 1

# define some constants
dimension_n = 100
hbar, mass 	= 1, 1
delta_x = (2 * l / dimension_n)

# this is the fraction in front
fractional_constant = ( (hbar **2) / (2 * mass * delta_x) )

# define function, this is the step_wise portion
def step_wise(i, i_prime):
	if(i == i_prime):
		return ( (np.pi**2) / 3 )
	else:
		return ( 2 / ((i - i_prime)**2) )


# this is the formula for constructing a single term T_i_iprime
def construct_single_term_in_kinetic_factor(i, i_prime):
	[fractional_constant * ((-1)**(i-i_prime)) * step_wise(i, i_prime) for i, i_prime in  zip(range(dimension_n), range(dimension_n))]


# operate on vectors of i, i_prime
construct_kinetic_factor = np.vectorize(construct_single_term_in_kinetic_factor)

# operate on vectors of 
construct_potential_factor = 