import numpy as np
import sys

# pycuda stuff
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.cumath as cumath
import pycuda.autoinit
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel

# scikit stuff
import skcuda.misc as misc
import skcuda.linalg as linalg
import skcuda.cublas as cublas

# Double precision is only supported by devices with compute capability >= 1.3:
gpu_data_type = np.float64 if misc.get_compute_capability(pycuda.autoinit.device) >= 1.3 else np.float32


np.random.seed(463912) # pick our seed for testing purposes

def diag_cpu(A, v1):
    # modelling the algorithm given by wikipedia
    m = A.shape[0]
    Q = np.ones((m, m), dtype=np.float64)
    Q[:, 0] = 0.0
    Q[:, 1] = v1.copy()
    beta = np.zeros(m, dtype=np.float64)
    alpha = np.zeros(m, dtype=np.float64) 

    for i in range(1, m-1):
        wp = np.dot(A, Q[:, i])
        alpha[i] = np.dot(wp, Q[:, i])
        w = wp - alpha[i]*Q[:, i] - beta[i]*Q[:, i-1]
        beta[i+1] = np.linalg.norm(w, ord=2)
        Q[:, i+1] = w / beta[i+1]

    wm = np.dot(A, Q[:, -1])

    alpha[-1] = np.dot(wm, Q[:, -1])
    print("CPU: ", alpha, beta, sep="\n\n")
    # make tridiagonal matrix out of alpha and beta
    # Tri = np.zeros(matrix_size)
    return


def diag_gpu(A, v1):
    # handle
    current_handle = cublas.cublasCreate()

    m = A.shape[0]
    Q = np.zeros((m, m), dtype=np.float64)
    # Q[0, :] = 0.0 # implied
    Q[1, :] = v1.copy()
    beta = np.zeros(m, dtype=np.float64)
    alpha = np.zeros(m, dtype=np.float64) 

    # move data onto the GPU
    A_gpu = gpuarray.to_gpu(A)
    Q_gpu = gpuarray.to_gpu(Q)
    beta_gpu = gpuarray.to_gpu(beta)
    alpha_gpu = gpuarray.to_gpu(alpha)
    w = gpuarray.zeros(m, dtype=np.float64)

    # we define three kernels for simple arithmetic
    w_scale = ElementwiseKernel(
        arguments="double *w, double *alpha, double *beta, double *Q1, double *Q2, int loop_index",
        operation="w[i] = w[i] - (alpha[loop_index] * Q1[i]) - (beta[loop_index] * Q2[i])",
        name="element_wise_w_building")
    # using -= to do inplace subtraction gives an incorrect answer


    norm_krnl = ReductionKernel(np.float64, neutral="0.0", reduce_expr="a+b", 
        map_expr="x[i]*x[i]", arguments="double *x")

    ediv = ElementwiseKernel(
        arguments="double *a, double *b, double *c, int loop_index",
        operation="a[i] = b[i] / c[loop_index+1]",
        name="element_wise_division")
    # the name must not have spaces!!!!

    for i in range(1, m-1):
        cublas.cublasDgemv(handle = current_handle, trans = 'T',
                            m = m, n = m, # Hermitian matrix
                            alpha = 1.0, 
                            beta = 0.0,
                            A = A_gpu.gpudata, 
                            lda = m,
                            x = Q_gpu[i, :].gpudata, 
                            incx = 1,    
                            y = w.gpudata, 
                            incy = 1,
                            )

        cublas.cublasDgemm(handle = current_handle, 
                            transa = 'n', transb = 'n',
                            m   = 1, n  = 1,      k = m,
                            lda = 1, ldb = m, ldc = 1,
                            alpha = 1.0,  beta = 0.0, 
                            A = w.gpudata, 
                            B = Q_gpu[i, :].gpudata, 
                            C = alpha_gpu[i].gpudata)


        w_scale(w, alpha_gpu, beta_gpu, Q_gpu[i, :], Q_gpu[i-1, :], i)
        beta_gpu[i+1] = cumath.sqrt(norm_krnl(w))
        ediv(Q_gpu[i+1, :], w, beta_gpu, i)
    # end of loop

    # last 2 steps
    cublas.cublasDgemv(handle = current_handle, trans = 'T',
                            m = m, n = m, # Hermitian matrix
                            alpha = 1.0,
                            beta = 0.0,
                            A = A_gpu.gpudata, 
                            lda = m,
                            x = Q_gpu[-1, :].gpudata,
                            incx = 1,    
                            y = w.gpudata,
                            incy = 1,)

    cublas.cublasDgemm(handle = current_handle, 
                        transa = 'n', transb = 'n',
                        m   = 1, n  = 1,  k = m,
                        lda = 1, ldb = m, ldc = 1,
                        alpha = 1.0,  beta = 0.0, 
                        A = w.gpudata, 
                        B = Q_gpu[-1, :].gpudata, 
                        C = alpha_gpu[-1].gpudata)

    # retrive the alpha's and betas
    alpha_cpu = alpha_gpu.get()
    beta_cpu = beta_gpu.get()

    print("GPU: ", alpha_cpu, beta_cpu, sep="\n\n")
    # make tridiagonal matrix out of alpha and B
    # Tri = np.zeros(matrix_size)
    return



if (__name__ == "__main__"):
    # input parameters
    matrix_size = int(sys.argv[1])

    # the matrix we will attempt to diagonalize
    matrix_A = np.random.random_sample((matrix_size, matrix_size))

    # the inital random vector
    vec_1 = np.random.random_sample(matrix_size)

    diag_gpu(matrix_A, vec_1)
    diag_cpu(matrix_A, vec_1)
