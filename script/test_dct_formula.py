import numpy as np

PI = np.pi


def get_dct_mats(M, N):
    dct_mat = np.zeros((M * N, M * N))
    dct_mat_inv = np.zeros((M * N, M * N))
    for row in range(M * N):
        i_prime, j_prime = divmod(row, N)
        for col in range(M * N):
            i, j = divmod(col, N)
            dct_mat[row, col] = np.cos(PI * (i + 0.5) * i_prime / M) * np.cos(
                PI * (j + 0.5) * j_prime / N
            )
            dct_mat_inv[row, col] = (
                4.0
                / (M * N)
                * np.cos(PI * (i_prime + 0.5) * i / M)
                * np.cos(PI * (j_prime + 0.5) * j / N)
            )
            if i == 0:
                dct_mat_inv[row, col] /= 2
            if j == 0:
                dct_mat_inv[row, col] /= 2
    return dct_mat, dct_mat_inv


def dct_forward(M, N, x):
    x_2d = x.reshape((M, N))
    v = np.zeros((M, N))
    v[: (M + 1) // 2, : (N + 1) // 2] = x_2d[::2, ::2]
    v[(M + 1) // 2 :, : (N + 1) // 2] = x_2d[(-1 - M % 2) :: -2, ::2]
    v[: (M + 1) // 2, (N + 1) // 2 :] = x_2d[::2, (-1 - N % 2) :: -2]
    v[(M + 1) // 2 :, (N + 1) // 2 :] = x_2d[(-1 - M % 2) :: -2, (-1 - N % 2) :: -2]
    v_hat = np.fft.fft2(v)
    x_2d_hat = np.copy(v_hat)
    for i_prime in range(M):
        i_theta = PI * i_prime / (2 * M)
        for j_prime in range(N):
            j_theta = PI * j_prime / (2 * N)
            if j_prime != 0:
                x_2d_hat[i_prime, j_prime] = (
                    0.5
                    * np.exp(-1.0j * i_theta)
                    * (
                        np.exp(-1.0j * j_theta) * v_hat[i_prime, j_prime]
                        + np.exp(1.0j * j_theta) * v_hat[i_prime, N - j_prime]
                    )
                )
            else:
                x_2d_hat[i_prime, 0] = np.exp(-1.0j * i_theta) * v_hat[i_prime, 0]
    x_2d_hat = np.real(x_2d_hat)
    x_hat = x_2d_hat.reshape((-1))
    return x_hat


def dct_backward(M, N, x_hat):
    x_2d_hat = x_hat.reshape((M, N))
    v_hat = np.zeros((M, N), dtype=np.cdouble)
    v_hat[:, :] = x_2d_hat[:, :]
    v_hat[1:, 1:] -= x_2d_hat[-1:0:-1, -1:0:-1]
    v_hat[1:, :] -= 1.0j * x_2d_hat[-1:0:-1, :]
    v_hat[:, 1:] -= 1.0j * x_2d_hat[:, -1:0:-1]
    for i_prime in range(M):
        i_theta = PI * i_prime / (2 * M)
        for j_prime in range(N):
            j_theta = PI * j_prime / (2 * N)
            v_hat[i_prime, j_prime] *= np.exp(1.0j * (i_theta + j_theta))
    v = np.fft.ifft2(v_hat)
    v = np.real(v)
    x_2d = np.zeros((M, N))
    x_2d[::2, ::2] = v[: (M + 1) // 2, : (N + 1) // 2]
    x_2d[::2, 1::2] = v[: (M + 1) // 2, -1 : (N + 1) // 2 - 1 : -1]
    x_2d[1::2, ::2] = v[-1 : (M + 1) // 2 - 1 : -1, : (N + 1) // 2]
    x_2d[1::2, 1::2] = v[-1 : (M + 1) // 2 - 1 : -1, -1 : (N + 1) // 2 - 1 : -1]
    x = x_2d.reshape((-1))
    return x


M, N = 7, 5
mat, mat_inv = get_dct_mats(M, N)
diag = np.ones((M, N))
diag[0, :] *= 0.5
diag[:, 0] *= 0.5
diag_mat = np.diag(diag.reshape((-1)))

iden_mat = np.eye(M * N)
print(np.linalg.norm(mat @ mat_inv - iden_mat))
print(np.linalg.norm(4.0 / (M * N) * mat.T @ diag_mat @ mat - iden_mat))

x = np.random.rand(M * N)
x_hat = dct_forward(M, N, x)
print(np.linalg.norm(mat @ x - x_hat))
x_ = dct_backward(M, N, x_hat)
print(np.linalg.norm(mat_inv @ x_hat - x_))
