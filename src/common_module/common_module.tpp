template int getCsrMatData<float>(std::vector<int> &csrRowOffsets, std::vector<int> &csrColInd, std::vector<float> &csrValues, const std::vector<int> &dims, const std::vector<double> &k_x, const std::vector<double> &k_y, const std::vector<double> &k_z);

template int getCsrMatData<double>(std::vector<int> &csrRowOffsets, std::vector<int> &csrColInd, std::vector<double> &csrValues, const std::vector<int> &dims, const std::vector<double> &k_x, const std::vector<double> &k_y, const std::vector<double> &k_z);

template int getStdRhsVec<float>(std::vector<float> &rhs, const std::vector<int> &dims, const std::vector<double> &k_z, const double delta_p);

template int getStdRhsVec<double>(std::vector<double> &rhs, const std::vector<int> &dims, const std::vector<double> &k_z, const double delta_p);

template int getHomoCoeffZ<float>(float &homoCoeffZ, const std::vector<float> &p, const std::vector<int> &dims, const std::vector<double> &k_z, const double delta_p, const double lenZ);

template int getHomoCoeffZ<double>(double &homoCoeffZ, const std::vector<double> &p, const std::vector<int> &dims, const std::vector<double> &k_z, const double delta_p, const double lenZ);