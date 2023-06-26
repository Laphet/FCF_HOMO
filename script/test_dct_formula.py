import numpy as np

PI = np.pi


def get_dct_mats(M, N):
    dct_mat = np.zeros((M * N, M * N))
    dct_mat_inv = np.zeros((M * N, M * N))
    for row in range(M * N):
        i_prime, j_prime = divmod(row, N)
        for col in range(M * N):
            i, j = divmod(col, N)
            dct_mat[row, col] = np.cos(PI * (i+0.5) * i_prime / M) * \
                np.cos(PI * (j+0.5) * j_prime / N)
            dct_mat_inv[row, col] = 4.0 / (M * N) * \
                np.cos(PI * (i_prime+0.5) * i / M) * \
                np.cos(PI * (j_prime+0.5) * j / N)
            if i == 0:
                dct_mat_inv[row, col] /= 2
            if j == 0:
                dct_mat_inv[row, col] /= 2
    return dct_mat, dct_mat_inv


def dct_forward(M, N, x):
    x_2d = x.reshape((M, N))
    v = np.zeros((M, N))
    v[0:(M+1)//2, 0:(N+1)//2] = x_2d[0::2, 0::2]
    v[(M+1)//2:, 0:(N+1)//2] = x_2d[(M - M % 2):0:-2, 0::2]
    v[0:(M+1)//2, (N+1)//2:] = x_2d[0::2, (N - N % 2):0:-2]
    v[(M+1)//2:, (N+1)//2:] = x_2d[(M - M % 2):0:-2, (N-N % 2):0:-2]
    v_hat = np.fft(v)
    x_2d_hat = np.copy(v_hat)
    for i_prime in range(M):
        i_theta = PI * i_prime / (2 * M)
        for j_prime in range(N):
            j_theta = PI * j_prime / (2 * N)
            if j_prime != 0:
                x_2d_hat[i_prime, j_prime] = 0.5 * np.exp(-1.0J * i_theta) \
                    * (np.exp(-1.0J * j_theta) * v_hat[i_prime, j_prime] +
                       np.exp(1.0J * j_theta) * v_hat[i_prime, N - j_prime])
            else:
                x_2d_hat[i_prime, 0] = 0.5 * np.exp(-1.0J * i_theta) \
                    * np.exp(-1.0J * j_theta) * v_hat[i_prime, 0]
    x_2d_hat = np.real(x_2d_hat)
    x_hat = x_2d_hat.reshape((-1))
    return x_hat


M, N = 4, 8
mat, mat_inv = get_dct_mats(M, N)
iden_mat = np.eye(M * N)
print(np.linalg.norm(mat @ mat_inv - iden_mat))
