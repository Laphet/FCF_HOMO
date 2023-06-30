template <>
void fctForward(float *out_hat, float const *in, float *realBuffer, cuda::std::complex<float> *compBuffer, cufftHandle plan, const int M, const int N, const int P);

template <>
void fctForward(double *out_hat, double const *in, double *realBuffer, cuda::std::complex<double> *compBuffer, cufftHandle plan, const int M, const int N, const int P);

template <>
void fctBackward(float *out, float const *in_hat, float *realBuffer, cuda::std::complex<float> *compBuffer, cufftHandle plan, const int M, const int N, const int P);

template <>
void fctBackward(double *out, double const *in_hat, double *realBuffer, cuda::std::complex<double> *compBuffer, cufftHandle plan, const int M, const int N, const int P);