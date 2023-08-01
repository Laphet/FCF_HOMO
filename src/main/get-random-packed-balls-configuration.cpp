#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

constexpr int    N{512};
constexpr double h{1.0 / N};

void get_bin_k(std::vector<int> &k_bin, const std::string &filename, const int num)
{
  std::vector<double> ballData(4 * num);
  std::ifstream       binFileReader(filename, std::ios::in | std::ios::binary);
  binFileReader.read(reinterpret_cast<char *>(&ballData[0]), num * sizeof(double));
  binFileReader.close();
#pragma omp parallel for
  for (int idx{0}; idx < num; ++idx) {
    double ball_x{ballData[4 * idx]}, ball_y{ballData[4 * idx + 1]}, ball_z{ballData[4 * idx + 2]};
    double ball_r{ballData[4 * idx + 3]};
    int    lattice_x = static_cast<int>(ball_x / h);
    int    lattice_y = static_cast<int>(ball_y / h);
    int    lattice_z = static_cast<int>(ball_z / h);
    int    lattice_r = static_cast<int>(ball_r / h) + 1;
    double x{0.0}, y{0.0}, z{0.0}, distance{0.0};
    for (int i{std::max(0, lattice_x - lattice_r)}; i < std::min(N, lattice_x + lattice_r + 1); ++i) {
      x = (i + 0.5) / N;
      for (int j{std::max(0, lattice_y - lattice_r)}; j < std::min(N, lattice_y + lattice_r + 1); ++j) {
        y = (j + 0.5) / N;
        for (int k{std::max(0, lattice_z - lattice_r)}; k < std::min(N, lattice_z + lattice_r + 1); ++k) {
          z        = (k + 0.5) / N;
          distance = (x - ball_x) * (x - ball_x);
          distance += (y - ball_y) * (y - ball_y);
          distance += (z - ball_z) * (z - ball_z);
          distance                     = std::sqrt(distance);
          k_bin[i * N * N + j * N + k] = distance <= ball_r ? 1 : 1.0;
        }
      }
    }
  }
}

int main(int argc, char *argv[])
{
  size_t           size = N * N * N;
  std::vector<int> k_bin(size);
  std::ofstream    binFileWriter;
  int              ballNum{-1};

  std::fill(k_bin.begin(), k_bin.end(), 0);
  ballNum = 55;
  get_bin_k(k_bin, std::string("bin/ball-list-55-r3.bin"), ballNum);
  binFileWriter.open("bin/k-55-r3.bin", std::ios::out | std::ios::binary);
  binFileWriter.write(reinterpret_cast<char *>(&k_bin[0]), size * sizeof(int));
  binFileWriter.close();

  std::fill(k_bin.begin(), k_bin.end(), 0);
  ballNum = 258;
  get_bin_k(k_bin, std::string("bin/ball-list-258-r2.bin"), ballNum);
  binFileWriter.open("bin/k-258-r2.bin", std::ios::out | std::ios::binary);
  binFileWriter.write(reinterpret_cast<char *>(&k_bin[0]), size * sizeof(int));
  binFileWriter.close();

  std::fill(k_bin.begin(), k_bin.end(), 0);
  ballNum = 1716;
  get_bin_k(k_bin, std::string("bin/ball-list-1716-r1.bin"), ballNum);
  binFileWriter.open("bin/k-1716-r1.bin", std::ios::out | std::ios::binary);
  binFileWriter.write(reinterpret_cast<char *>(&k_bin[0]), size * sizeof(int));
  binFileWriter.close();

  return EXIT_SUCCESS;
}
