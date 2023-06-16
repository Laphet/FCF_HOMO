### Data description
The domain is discretized into MxNxP cells in x-, y- and z-direction.
For a 3d index “(i, j, k)” with “0 <= i < M”, “0 <= j < N” and “0 <= k < P”,
we should convert it into a 1d index as “(ixNxP + jxP + k)”.

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





### FFTW implementation