#include "../include/spmm.h"
#include "../include/utils.h"
#include <iostream>
#include <string>
#include <taco.h>
#include <cmath>
#include <vector>
#include <chrono>

using namespace taco;

Tensor<double> softmax(const Tensor<double> &x, Format outFormat) {
    Tensor<double> sum({x.getDimension(0)}, Format({dense}));
    Tensor<double> out({x.getDimension(0), x.getDimension(1)}, outFormat);
    IndexVar i, j;

    sum(i) = exp(x(i, j));
    out(i, j) = exp(x(i, j)) / sum(i);

    return out;
}

const float samplingTaco(const Tensor<double> &input, int xStride, int yStride) {
    int rows = input.getDimension(0);
    int cols = input.getDimension(1);
    int count = 0;

    for (auto &val : input)
      if (val.first[0] % xStride == 0 && val.first[1] % yStride == 0)
          if (val.second == 0)
            count++;

    int xTotal = (cols + xStride - 1) / xStride;
    int yTotal = (rows + yStride - 1) / yStride;
    return static_cast<double>(count) / (xTotal * yTotal);
}

bool allClose(const std::vector<std::vector<double>> &mat, Tensor<double> tensor) {
    double eps = 0.00001;
    for (auto &val : tensor)
      if (std::abs(mat[val.first[0]][val.first[1]] - val.second) > eps) {
          std::cout << mat[val.first[0]][val.first[1]] << std::endl;
          std::cout << val.second << std::endl;
          return false;
      }
    return true;
}

std::vector<std::vector<double>> cppSelfAttention(const std::vector<std::vector<double>> &Q, const std::vector<std::vector<double>> &K, const std::vector<std::vector<double>> &V) {
    int n_q = Q.size(), d_k = Q[0].size();
    int n_k = V.size(), d_v = V[0].size();
    std::vector<std::vector<double>> P(n_q, std::vector<double>(n_k, 0.0));
    float normalizer = std::sqrt(d_k);

    std::cout << (Q[0].size() == K[0].size()) <<std::endl;

    for (int i = 0; i < n_q; i++)
        for (int j = 0; j < n_k; j++)
            for (int k = 0; k < d_k; k++)
                P[i][j] += Q[i][k] * K[j][k] / normalizer;


    std::vector<double> sum(n_q, 0.0);
    for (int i = 0; i < n_q; ++i)
        for (int j = 0; j < n_k; ++j)
            sum[i] += std::exp(P[i][j]);

    for (int i = 0; i < n_q; ++i)
        for (int j = 0; j < n_k; ++j)
            P[i][j] = std::exp(P[i][j]) / sum[i];

    std::vector<std::vector<double>> O(n_q, std::vector<double>(d_v, 0.0));
    for (int i = 0; i < n_q; i++)
        for (int j = 0; j < d_v; j++)
            for (int k = 0; k < n_k; k++)
                O[i][j] += P[i][k] * V[k][j];

    return O;
}

Tensor<double> selfAttention(const Tensor<double> &Q, const Tensor<double> &K, const Tensor<double> &V) {
    int n_q = Q.getDimension(0), d_k = Q.getDimension(1);
    int n_k = V.getDimension(0), d_v = V.getDimension(1);
    /*Tensor<double> Kt = K.transpose({1, 0});*/
    float normalizer = std::sqrt(d_k);
    Tensor<double> prod({n_q, n_k}, {Dense, Dense});
    Tensor<double> P({n_q, n_k}, {Dense, Dense});
    Tensor<double> O({n_q, d_v}, {Dense, Dense});
    IndexVar i, j, k;
    
    /*std::cout << Q.getStorage().getFormat() << std::endl;*/
    /*std::cout << K.getStorage().getFormat() << std::endl;*/
    prod(i, j) = Q(i, k) * K(j, k); // * MASK (can sample)
    /*prod(i, j) = Q(i, k) * Kt(k, j); // * MASK (can sample)*/
    P(i, j) = prod(i, j) / normalizer; // elementwise, can sample
    P = softmax(P, {Dense, Dense}); // elementwise, can sample
    O(i, j) = P(i, k) * V(k, j); // matmul, can sample

    O.evaluate();

    return O;
}

void run(int n_q, int n_k, int d_k, int d_v, double sparsity, Format QFormat, Format KFormat, Format VFormat, bool sample, bool convert) {
    auto Qc = genMatrix(n_q, d_k, sparsity);
    auto Kc = genMatrix(n_k, d_k, sparsity);
    auto Vc = genMatrix(n_k, d_v, sparsity);
    Tensor<double> Q, K, V;
    if (!convert) {
        Q = convertToTACO(Qc, QFormat);
        K = convertToTACO(Kc, KFormat);
        V = convertToTACO(Vc, VFormat);
    }
    
    const auto start{std::chrono::steady_clock::now()};
    if (sample) {
        sampling(Qc, sparsity, false, 64, 64);
        sampling(Kc, sparsity, false, 64, 64);
        sampling(Vc, sparsity, false, 64, 64);
    }
    if (convert) {
        Q = convertToTACO(Qc, QFormat);
        K = convertToTACO(Kc, KFormat);
        V = convertToTACO(Vc, VFormat);
    }
    selfAttention(Q, K, V);
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    /*std::cout << allClose(cppSelfAttention(Qc, Kc, Vc), O) << std::endl;*/
    std::cout << elapsed_seconds.count() << std::endl; 
}

Format stringToFormat(std::string formatString) {
    Format dense({Dense, Dense});
    Format csr({Dense, Sparse});
    Format csc({Dense, Sparse}, {1, 0});

    if (formatString == "DD")
        return dense;
    if (formatString == "CSR")
        return csr;
    if (formatString == "CSC")
        return csc;
    std::cerr << "Invalid tensor format" << std::endl;
    std::exit(1);
}

int parseArgumentsAndRun(int argc, char *argv[]) {
    if (argc != 11) {
        std::cerr << "Usage: " << argv[0]
            << " <n_q> <n_k> <d_k> <d_v> <sparsity> <QFormat> <KFormat> <VFormat> <sample (0 or 1)> <convert (0 or 1)>\n";
        return 1;
    }

    int n_q = std::stoi(argv[1]);
    int n_k = std::stoi(argv[2]);
    int d_k = std::stoi(argv[3]);
    int d_v = std::stoi(argv[4]);
    double sparsity = std::stod(argv[5]); // assume all are the same level of sparsity
    Format QFormat = stringToFormat(argv[6]);
    Format KFormat = stringToFormat(argv[7]);
    Format VFormat = stringToFormat(argv[8]);
    bool sample = std::stoi(argv[9]);
    bool convert = std::stoi(argv[10]);

    run(n_q, n_k, d_k, d_v, sparsity, QFormat, KFormat, VFormat, sample, convert);
    return 0;
}

void runTests(int argc, char *argv[]) { parseArgumentsAndRun(argc, argv); }
