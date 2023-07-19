import numpy as np
from scipy.optimize import linprog

VALS_LENGTH = 5


def get_optimal_kref(k_maxmin_vals):
    # The vector c is in the objective function.
    # The order of decision variable is: log_k_x, log_k_y, log_k_z,
    # log_k_in, log_k_out, log_lambda, log_Lambda
    c = np.zeros((VALS_LENGTH + 2,))
    c[-1] = 1.0
    c[-2] = -1.0
    # The vector b is the rhs of the inequality system.
    b = np.zeros((VALS_LENGTH * 2,))
    # The matrix A is the operator of the inequality system.
    A = np.zeros((VALS_LENGTH * 2, VALS_LENGTH + 2))
    for i in range(VALS_LENGTH):
        b[2 * i] = -np.log(k_maxmin_vals[i])
        A[2 * i, -1] = -1.0
        A[2 * i, i] = -1.0
        b[2 * i + 1] = np.log(k_maxmin_vals[i + VALS_LENGTH])
        A[2 * i + 1, -2] = 1.0
        A[2 * i + 1, i] = 1.0
    res = linprog(c=c, A_ub=A, b_ub=b, bounds=(None, None))
    print(res.message)
    print("The optimal value={0:.6e}".format(np.exp(res.fun)))
    return np.exp(res.x)


if __name__ == "__main__":
    filename = "k-maxmin-vals.bin"
    print("Python script reads [{0:s}]".format(filename))
    kvals = np.fromfile("bin/" + filename, dtype=np.float64)
    optimal_kref = get_optimal_kref(kvals)
    print(optimal_kref)
    filename = "k-ref-vals.bin"
    print("Python script writes [{0:s}]".format(filename))
    optimal_kref.tofile("bin/" + filename)
