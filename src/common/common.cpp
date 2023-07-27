#include "common.hpp"

int getIdxFrom3dIdx(const int i, const int j, const int k, const int N, const int P)
{
  return i * N * P + j * P + k;
}

void get3dIdxFromIdx(int &i, int &j, int &k, const int idx, const int N, const int P)
{
  i = idx / (N * P);
  j = (idx / P) % N;
  k = idx % P;
}

template <typename T>
void common<T>::analysisCoeff(const std::vector<double> &k_x, const std::vector<double> &k_y, const std::vector<double> &k_z, std::vector<double> &k_vals)
{
  constexpr int VALS_LENGTH{5};
  int           M{dims[0]}, N{dims[1]}, P{dims[2]};
  int           size{M * N * P};
  double        kmax[]{0.0, 0.0, 0.0, 0.0, 0.0};
  double        doubleMax{std::numeric_limits<double>::max()};
  double        kmin[]{doubleMax, doubleMax, doubleMax, doubleMax, doubleMax};

#pragma omp parallel for reduction(max : kmax[ : 5]) reduction(min : kmin[ : 5])
  for (int idx{0}; idx < size; ++idx) {
    int i{0}, j{0}, k{0};
    int adjIdx{0};
    get3dIdxFromIdx(i, j, k, idx, N, P);
    if (1 <= i) {
      adjIdx  = getIdxFrom3dIdx(i - 1, j, k, N, P);
      kmax[0] = std::max(kmax[0], 2.0 / (1.0 / k_x[idx] + 1.0 / k_x[adjIdx]));
      kmin[0] = std::min(kmin[0], 2.0 / (1.0 / k_x[idx] + 1.0 / k_x[adjIdx]));
    }
    if (1 <= j) {
      adjIdx  = getIdxFrom3dIdx(i, j - 1, k, N, P);
      kmax[1] = std::max(kmax[1], 2.0 / (1.0 / k_y[idx] + 1.0 / k_y[adjIdx]));
      kmin[1] = std::min(kmin[1], 2.0 / (1.0 / k_y[idx] + 1.0 / k_y[adjIdx]));
    }
    if (1 <= k) {
      adjIdx  = getIdxFrom3dIdx(i, j, k - 1, N, P);
      kmax[2] = std::max(kmax[2], 2.0 / (1.0 / k_z[idx] + 1.0 / k_z[adjIdx]));
      kmin[2] = std::min(kmin[2], 2.0 / (1.0 / k_z[idx] + 1.0 / k_z[adjIdx]));
    }
    if (0 == k) {
      kmax[3] = std::max(kmax[3], k_z[idx]);
      kmin[3] = std::min(kmin[3], k_z[idx]);
    }
    if (P - 1 == k) {
      kmax[4] = std::max(kmax[4], k_z[idx]);
      kmin[4] = std::min(kmin[4], k_z[idx]);
    }
  }

#ifdef DEBUG
  std::cout << "kmax=[";
  for (int i{0}; i < VALS_LENGTH; ++i) std::printf("%.6e, ", kmax[i]);
  std::cout << "]\n";
  std::cout << "kmin=[";
  for (int i{0}; i < VALS_LENGTH; ++i) std::printf("%.6e, ", kmin[i]);
  std::cout << "]\n";
#endif

  std::string   finename("bin/k-maxmin-vals.bin");
  std::ofstream binFileWriter(finename, std::ios::out | std::ios::binary);
  binFileWriter.write(reinterpret_cast<char *>(&kmax[0]), VALS_LENGTH * sizeof(double));
  binFileWriter.write(reinterpret_cast<char *>(&kmin[0]), VALS_LENGTH * sizeof(double));
  binFileWriter.close();

#ifdef DEBUG
  std::cout << "Write kmax/min values into [" << finename << "].\n";
  std::cout << "Please cd to the project root directory.\n";
#endif

  const std::string cmd = "python script/optimal_ref_parameters.py";
  std::system(cmd.c_str());

  finename = "bin/k-ref-vals.bin";

#ifdef DEBUG
  std::cout << "Read kref values from [" << finename << "].\n";
#endif

  std::ifstream binFileReader(finename, std::ios::in | std::ios::binary);
  char          buffer[1024];
  binFileReader.read(buffer, VALS_LENGTH * sizeof(double));
  binFileReader.close();

  std::memcpy(reinterpret_cast<void *>(&k_vals[0]), reinterpret_cast<void *>(&kmax[0]), VALS_LENGTH * sizeof(double));
  std::memcpy(reinterpret_cast<void *>(&k_vals[5]), reinterpret_cast<void *>(&kmin[0]), VALS_LENGTH * sizeof(double));
  std::memcpy(reinterpret_cast<void *>(&k_vals[10]), reinterpret_cast<void *>(&buffer[0]), VALS_LENGTH * sizeof(double));

#ifdef DEBUG
  std::cout << "kref=[";
  for (int i{0}; i < VALS_LENGTH; ++i) std::printf("%.6e, ", k_vals[i + 10]);
  std::cout << "]\n";
#endif
}

template <typename T>
void common<T>::getSprMatData(std::vector<int> &csrRowOffsets, std::vector<int> &csrColInd, std::vector<T> &csrValues, const std::vector<double> &k_x, const std::vector<double> &k_y, const std::vector<double> &k_z)
{
  int M{dims[0]}, N{dims[1]}, P{dims[2]};
  int size{M * N * P}, row{0};

  csrRowOffsets[0] = 0;
#pragma omp parallel for
  for (row = 0; row < size; ++row) {
    int    i{0}, j{0}, k{0}, col{0};
    double mean_k = 0.0;

    get3dIdxFromIdx(i, j, k, row, N, P);
    csrRowOffsets[row + 1] = 0;
    // cols order, 0, z-, z+, y-, y+, x-, x+
    csrColInd[row * STENCIL_WIDTH] = row;
    csrValues[row * STENCIL_WIDTH] = static_cast<T>(0.0);
    if (k - 1 >= 0) {
      col = getIdxFrom3dIdx(i, j, k - 1, N, P);
      // col    = i * N * P + j * P + k - 1;
      mean_k = 2.0 / (1.0 / k_z[row] + 1.0 / k_z[col]);
      csrValues[row * STENCIL_WIDTH] += static_cast<T>(mean_k);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      csrValues[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = static_cast<T>(-mean_k);
    }
    if (k + 1 < P) {
      col = getIdxFrom3dIdx(i, j, k + 1, N, P);
      // col    = i * N * P + j * P + k + 1;
      mean_k = 2.0 / (1.0 / k_z[row] + 1.0 / k_z[col]);
      csrValues[row * STENCIL_WIDTH] += static_cast<T>(mean_k);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      csrValues[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = static_cast<T>(-mean_k);
    }
    if (j - 1 >= 0) {
      col = getIdxFrom3dIdx(i, j - 1, k, N, P);
      // col    = i * N * P + (j - 1) * P + k;
      mean_k = 2.0 / (1.0 / k_y[row] + 1.0 / k_y[col]);
      csrValues[row * STENCIL_WIDTH] += static_cast<T>(mean_k);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      csrValues[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = static_cast<T>(-mean_k);
    }
    if (j + 1 < N) {
      col = getIdxFrom3dIdx(i, j + 1, k, N, P);
      // col    = i * N * P + (j + 1) * P + k;
      mean_k = 2.0 / (1.0 / k_y[row] + 1.0 / k_y[col]);
      csrValues[row * STENCIL_WIDTH] += static_cast<T>(mean_k);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      csrValues[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = static_cast<T>(-mean_k);
    }
    if (i - 1 >= 0) {
      col = getIdxFrom3dIdx(i - 1, j, k, N, P);
      // col    = (i - 1) * N * P + j * P + k;
      mean_k = 2.0 / (1.0 / k_x[row] + 1.0 / k_x[col]);
      csrValues[row * STENCIL_WIDTH] += static_cast<T>(mean_k);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      csrValues[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = static_cast<T>(-mean_k);
    }
    if (i + 1 < M) {
      col = getIdxFrom3dIdx(i + 1, j, k, N, P);
      // col    = (i + 1) * N * P + j * P + k;
      mean_k = 2.0 / (1.0 / k_x[row] + 1.0 / k_x[col]);
      csrValues[row * STENCIL_WIDTH] += static_cast<T>(mean_k);
      csrRowOffsets[row + 1]++;
      csrColInd[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = col;
      csrValues[row * STENCIL_WIDTH + csrRowOffsets[row + 1]] = static_cast<T>(-mean_k);
    }
    if (k == 0 || k == P - 1) csrValues[row * STENCIL_WIDTH] += static_cast<T>(2.0 * k_z[row]);
    csrRowOffsets[row + 1]++;
  }

  /* Clean the unused memory. */
  for (row = 0; row < size; ++row) {
    std::memmove(&csrColInd[csrRowOffsets[row]], &csrColInd[row * STENCIL_WIDTH], sizeof(int) * csrRowOffsets[row + 1]);
    std::memmove(&csrValues[csrRowOffsets[row]], &csrValues[row * STENCIL_WIDTH], sizeof(T) * csrRowOffsets[row + 1]);
    csrRowOffsets[row + 1] += csrRowOffsets[row];
  }

/* Now, csrValues[csrRowoffsets[i]] = diag_i. */
/* MKL needs in every row, the column index of the diagonal entry starts first. */
/* Sort column indexes. */
#pragma omp parallel for
  for (row = 0; row < size; ++row) {
    int nnzCurrentRow = csrRowOffsets[row + 1] - csrRowOffsets[row];
    for (int i{0}; i < nnzCurrentRow - 1; ++i)
      for (int j{1}; j < nnzCurrentRow - 1 - i; ++j) {
        if (csrColInd[csrRowOffsets[row] + j] > csrColInd[csrRowOffsets[row] + j + 1]) {
          std::swap(csrColInd[csrRowOffsets[row] + j], csrColInd[csrRowOffsets[row] + j + 1]);
          std::swap(csrValues[csrRowOffsets[row] + j], csrValues[csrRowOffsets[row] + j + 1]);
        }
      }
  }
}

template <typename T>
void common<T>::sortSprMatData(const std::vector<int> &csrRowOffsets, std::vector<int> &csrColInd, std::vector<T> &csrValues)
{
  /* cusparse's legacy incomplete Cholesky needs the column indices to be sorted. */
  size_t size = dims[0] * dims[1] * dims[2];
#pragma omp parallel for
  for (int row{0}; row < size; ++row) {
    int nnzCurrentRow = csrRowOffsets[row + 1] - csrRowOffsets[row];
    for (int i{0}; i < nnzCurrentRow; ++i)
      for (int j{0}; j < nnzCurrentRow - 1 - i; ++j) {
        if (csrColInd[csrRowOffsets[row] + j] > csrColInd[csrRowOffsets[row] + j + 1]) {
          std::swap(csrColInd[csrRowOffsets[row] + j], csrColInd[csrRowOffsets[row] + j + 1]);
          std::swap(csrValues[csrRowOffsets[row] + j], csrValues[csrRowOffsets[row] + j + 1]);
        }
      }
  }
}

template <typename T>
void common<T>::getStdRhsVec(std::vector<T> &rhs, const std::vector<double> &k_z, const double delta_p)
{
  int M{dims[0]}, N{dims[1]}, P{dims[2]};
  int size{M * N * P}, row{0};

#pragma omp parallel for
  for (int row{0}; row < size; ++row) {
    int i{0}, j{0}, k{0};
    get3dIdxFromIdx(i, j, k, row, N, P);
    rhs[row] = 0;
    if (k == 0) rhs[row] += 2 * static_cast<T>(k_z[row] * delta_p);
  }
}

template <typename T>
void common<T>::getHomoCoeffZ(T &homoCoeffZ, const std::vector<T> &p, const std::vector<double> &k_z, const double delta_p, const double lenZ)
{
  int    M{dims[0]}, N{dims[1]}, P{dims[2]}, i{0}, j{0}, row{0};
  double temp{0.0};

  homoCoeffZ = 0;
#pragma omp parallel for reduction(+ : homoCoeffZ)
  for (i = 0; i < M; ++i)
    for (j = 0; j < N; ++j) {
      row = getIdxFrom3dIdx(i, j, 0, N, P);
      // row  = i * N * P + j * P;
      temp = 2.0 * lenZ * lenZ / (M * N * P) * k_z[row] * (delta_p - static_cast<double>(p[row])) / delta_p;
      homoCoeffZ += static_cast<T>(temp);
    }
}

template <typename T>
void common<T>::getTridSolverData(std::vector<T> &dl, std::vector<T> &d, std::vector<T> &du, const std::vector<T> &homoParas)
{
  int M{dims[0]}, N{dims[1]}, P{dims[2]};
  T   myPi{static_cast<T>(M_PI)};
  T   k_x_ref{homoParas[0]}, k_y_ref{homoParas[1]}, k_z_ref{homoParas[2]};
  T   k_in_ref{homoParas[3]}, k_out_ref{homoParas[4]};
#pragma omp parallel for
  for (int matIdx{0}; matIdx < M * N; ++matIdx) {
    int i{matIdx / N}, j{matIdx % N};
    T   temp_i{2 * (1 - std::cos(static_cast<T>(i) / M * myPi))};
    T   temp_j{2 * (1 - std::cos(static_cast<T>(j) / N * myPi))};
    dl[matIdx * P] = 0;
    d[matIdx * P]  = (k_x_ref * temp_i) + (k_y_ref * temp_j) + k_z_ref + 2 * k_in_ref;
    du[matIdx * P] = -k_z_ref;
    for (int k = 1; k < P - 1; ++k) {
      dl[matIdx * P + k] = -k_z_ref;
      d[matIdx * P + k]  = (k_x_ref * temp_i) + (k_y_ref * temp_j) + (2 * k_z_ref);
      du[matIdx * P + k] = -k_z_ref;
    }
    dl[(matIdx + 1) * P - 1] = -k_z_ref;
    d[(matIdx + 1) * P - 1]  = (k_x_ref * temp_i) + (k_y_ref * temp_j) + k_z_ref + (2 * k_out_ref);
    du[(matIdx + 1) * P - 1] = 0;
  }
}

template <typename T>
void common<T>::getSsorData(const std::vector<int> &csrRowOffsets, const std::vector<int> &csrColInd, const std::vector<T> &csrValues, const T omega, std::vector<T> &ssorValues)
{
  if (!(static_cast<T>(0) < omega && omega < static_cast<T>(2))) {
    std::cerr << "omega=" << omega << " does not satisfy (0, 2)." << std::endl;
    std::cerr << "There is nothing to do in this routine." << std::endl;
    return;
  }

  size_t         size = dims[0] * dims[1] * dims[2];
  size_t         nnz  = csrRowOffsets[size];
  std::vector<T> diag(size);
  T              temp{1 / (2 - omega)};

#pragma omp parallel for
  for (int row{0}; row < size; ++row) diag[row] = csrValues[csrRowOffsets[row]];

#pragma omp parallel for
  for (int row{0}; row < size; ++row) {
    int col{0};
    for (int i{0}; i < csrRowOffsets[row + 1] - csrRowOffsets[row]; ++i) {
      col = csrColInd[csrRowOffsets[row] + i];
      /* Diagonal */
      if (col == row) ssorValues[csrRowOffsets[row] + i] = diag[row] * temp / omega;
      /* L part. */
      if (col < row) ssorValues[csrRowOffsets[row] + i] = csrValues[csrRowOffsets[row] + i] * omega / diag[col];
      /* U part. */
      if (col > row) ssorValues[csrRowOffsets[row] + i] = csrValues[csrRowOffsets[row] + i] * temp;
    }
  }
}

template <typename T>
void common<T>::setTestVecs(std::vector<T> &v, std::vector<T> &v_hat)
{
  int M{dims[0]}, N{dims[1]}, P{dims[2]};
  int size{M * N * P};
  int i{0}, j{0}, k{0}, i_t{1}, j_t{2};
  T   myPi{static_cast<T>(M_PI)}, myHalf{static_cast<T>(0.5)};
  for (int row{0}; row < size; ++row) {
    get3dIdxFromIdx(i, j, k, row, N, P);
    if (i_t == i && j_t == j) v_hat[row] = 1;
    else v_hat[row] = 0;
    v[row] = static_cast<T>(4) / static_cast<T>(M * N);
    v[row] *= std::cos(myPi * (static_cast<T>(i) + myHalf) * static_cast<T>(i_t) / static_cast<T>(M));
    v[row] *= std::cos(myPi * (static_cast<T>(j) + myHalf) * static_cast<T>(j_t) / static_cast<T>(N));
  }
}

template <typename T>
void common<T>::setTestForPrecondSolver(std::vector<T> &u, std::vector<T> &rhs, const T k_x, const T k_y, const T k_z)
{
  int M{dims[0]}, N{dims[1]}, P{dims[2]};
  T   h_x{static_cast<T>(1) / M}, h_y{static_cast<T>(1) / N}, h_z{static_cast<T>(1) / P};
  T   myHalf{static_cast<T>(0.5)}, myPi{static_cast<T>(M_PI)};
#pragma omp parallel for
  for (int idx{0}; idx < M * N * P; ++idx) {
    int i{0}, j{0}, k{0};
    get3dIdxFromIdx(i, j, k, idx, N, P);
    T x{(i + myHalf) * h_x}, y{(j + myHalf) * h_y}, z{(k + myHalf) * h_z};
    /* This is the solution. */
    u[idx]   = std::cos(x * myPi) * std::cos(y * myPi) * std::exp(z);
    rhs[idx] = k_x * myPi * myPi * std::cos(x * myPi) * std::cos(y * myPi) * std::exp(z);
    rhs[idx] += k_y * myPi * myPi * std::cos(x * myPi) * std::cos(y * myPi) * std::exp(z);
    rhs[idx] -= k_z * std::cos(x * myPi) * std::cos(y * myPi) * std::exp(z);
    if (0 == k) {
      z = 0;
      rhs[idx] += 2 * k_z * P * P * std::cos(x * myPi) * std::cos(y * myPi) * std::exp(z);
    }
    if (P - 1 == k) {
      z = 1;
      rhs[idx] += 2 * k_z * P * P * std::cos(x * myPi) * std::cos(y * myPi) * std::exp(z);
    }
  }
}

template <typename T>
void common<T>::setTestForSolver(std::vector<double> &k_x, std::vector<double> &k_y, std::vector<double> &k_z, std::vector<T> &u, std::vector<T> &rhs)
{
  int M{dims[0]}, N{dims[1]}, P{dims[2]};
  T   h_x{static_cast<T>(1) / M}, h_y{static_cast<T>(1) / N}, h_z{static_cast<T>(1) / P};
  T   myHalf{static_cast<T>(0.5)}, myPi{static_cast<T>(M_PI)};
#pragma omp parallel for
  for (int idx{0}; idx < M * N * P; ++idx) {
    int i{0}, j{0}, k{0};
    get3dIdxFromIdx(i, j, k, idx, N, P);
    T      x{(i + myHalf) * h_x}, y{(j + myHalf) * h_y}, z{(k + myHalf) * h_z};
    double x_d{(i + 0.5) / M}, y_d{(j + 0.5) / N}, z_d{(k + 0.5) / P};
    /* This is the solution. */
    u[idx] = std::cos(x * myPi) * std::cos(y * myPi) * std::exp(z);
    /* Those are the coefficients. */
    k_x[idx] = (std::cos(y_d * M_PI) + 2) * M * M;
    k_y[idx] = (2 * std::exp(z_d)) * N * N;
    k_z[idx] = (3 * std::cos(x_d * M_PI) + 4) * P * P;
    /* f = Exp[z]  Cos[Pi x] Cos[Pi y] (2 (-2 + Pi^2 + Exp[z] Pi^2) - 3 Cos[Pi x] + Pi^2 Cos[Pi y]) */
    rhs[idx] = u[idx];
    rhs[idx] *= 2 * (-2 + myPi * myPi + std::exp(z) * myPi * myPi) - 3 * std::cos(myPi * x) + myPi * myPi * std::cos(myPi * y);
    if (0 == k) {
      z = static_cast<T>(0);
      rhs[idx] += 2 * k_z[idx] * std::cos(x * myPi) * std::cos(y * myPi) * std::exp(z);
    }
    if (P - 1 == k) {
      z = static_cast<T>(1);
      rhs[idx] += 2 * k_z[idx] * std::cos(x * myPi) * std::cos(y * myPi) * std::exp(z);
    }
  }
}

template struct common<float>;

template struct common<double>;

inputParser::inputParser(int &argc, char **argv)
{
  for (int i = 1; i < argc; ++i) this->tokens.push_back(std::string(argv[i]));
}

const std::string &inputParser::getCmdOption(const std::string &option) const
{
  std::vector<std::string>::const_iterator itr;
  itr = std::find(this->tokens.begin(), this->tokens.end(), option);
  if (itr != this->tokens.end() && ++itr != this->tokens.end()) { return *itr; }
  static const std::string empty_string("");
  return empty_string;
}

bool inputParser::cmdOptionExists(const std::string &option) const
{
  return std::find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end();
}
