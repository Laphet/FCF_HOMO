### Data description
The domain is discretized into "MxNxP" cells in x-, y- and z-direction.
For a 3d index “(i, j, k)” with “0 <= i < M”, “0 <= j < N” and “0 <= k < P”,
we should convert it into a 1d index as “(ixNxP + jxP + k)”, i.e.,
``k'' goes the fastest.

Input three 1d arrays---“k_x”, “k_y” and “k_z”---with the described data layout.
We should normalize “k_x”, “k_y” and “k_z” by “k_x <- k_x/h_x^2”, “k_y <- k_y/h_y^2” 
and “k_z <- k_z/h_z^2” beforehand.

Construct the sparse linear system “Ax = b”, where “A” is a sparse matrix in the CSR format.
The right-hand vector “b” may come from the homogenization setting or arbitrary conditions.

We will focus on the 0-based index.

The data type of “A” and “b” could be **double** or **float**.


### CUDA implementation
To construct a CSR sparse matrix, we may need a routine:
- cusparseCreateCsr().

To destroy a CSR sparse matrix, we may need a routine:
- cusparseDestroySpMat().

To calculate the product of a sparse matrix and a dense vector, we may need a routine:
- cusparseSpMV().

Note that we will also use this routine to obtain the residual.

The cuda tridiagonal solver is included in cuSparse.
We will use the batched tridiagonal solver to maximize efficiency.
- cusparse<S, D, C, Z>gtsv2StridedBatch_bufferSizeExt(),
- cusparse<S, D, C, Z>gtsv2StridedBatch(),
the RHS will be overwritten.

Note that there is another batched tridiagonal solver called
- cusparse<S, D, C, Z>gtsvInterleavedBatch(),

which needs a different data layout ("k" goes the slowest) compared with the former one.

Due to that cuFFT does not implement a fast cosine transformation,
we need to apply a special treatment (see reference 1).
Note that we need to write kernel functions to implement pre- and post-processes. 
The formula in the original reference contains a typo, 
and check the Python script for the correct one.

We focus on two types of transformations:
- R2C, D2Z,
- C2R, Z2D.

We need the following routines from cuFFT:
- cufftCreate(),
- cufftMakePlanMany(),
- cufftExecR2C(), cufftExecD2Z(),
- cufftExecC2R(), cufftExecZ2D(),
- cufftDestroy().

### FFTW and oneapi implementation
Use the following commands to configure fftw:
CC=icx ./configure --prefix=${HOME}/fftw3 --enable-avx2 --enable-avx512 --enable-openmp --enable-threads

To enable using fftw3 in the float precision, we need to reconfigure
CC=icx ./configure --prefix=${HOME}/fftw3 --enable-single --enable-avx2 --enable-avx512 --enable-openmp --enable-threads

Note that the original fftw3.pc (fftw3f.pc) is not for linking to the OpenMP version,
and we use the modified .pc file instead.

We will use the following routines:
- fftw_malloc();
- fftw_free();
- fftw_plan_many_r2r(), different types for the forward and backward directions;
- fftw_execute();
- fftw_execute_r2r(), use the same plan;
- fftw_destroy_plan();
- fftw_cleanup().

The tridiagonal solver in mkl uses
- LAPACKE_<s,d>pttrf()
to factorize the matrix first (the diagonal and sub-diagonal parts will be overwritten).
Then solve the system by
- LAPACKE_<s,d>pttrs(),
and the RHS vector will be overwritten.


### Set up development environment
- Install oneapi in the home directory, and use oneapi.config to (only) enable compilers (icx, icpx).
- Install cmake via conda.
- Install clang-format via conda.
- Change vscode cmake tools settings.
- Move .pc files to ./externals.

### Use valgrind to detect memory leaks
Install valgrind via conda, and try:

- valgrind --leak-check=yes ./main
- valgrind --leak-check=full --show-leak-kinds=all ./main > valgrind_report.txt

### References
[1] Makhoul, J. (1980). A fast cosine transform in one and two dimensions. 
IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(1), 27-34.