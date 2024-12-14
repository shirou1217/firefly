#include <algorithm>
#include <chrono>
#include <cmath>
#include <emmintrin.h>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <omp.h>
#include <random>
#include <vector>
// #include "/home/pp24/pp24s036/firefly/NVTX/c/include/nvtx3/nvtx3.hpp"

using namespace std;
const double ten = 10.0;
const double two_pi = 2.0 * M_PI;
__m512d vec_ten = _mm512_set1_pd(ten);
__m512d vec_two_pi = _mm512_set1_pd(two_pi);
vector<double> fitness, pop;
int N, D;
int num_threads;

class FA {
  public:
    FA(int dimen, int population, int max_iter)
        : D(dimen), N(population), it(max_iter), A(0.97), B(1.0), G(0.0001) {
        // nvtxRangePushA("FA() initialize parameter");
        Ub.resize(D, 3.0);
        Lb.resize(D, 0.0);
        // nvtxRangePop();
    }

    void fun() {
        fitness.assign(N, 10 * D);

#pragma omp parallel for collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < D; j++) {
                double x = pop[i * D + j];
                fitness[i] += x * x - 10 * cos(2 * M_PI * x);
            }
        }
    }
    void fun2() {
        fitness.assign(N, 10 * D);
#pragma omp parallel for collapse(2) num_threads(num_threads)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < D; j += 8) {
                int remaining = D - j;
                __mmask8 mask = (remaining >= 8) ? 0xFF : (1 << remaining) - 1;
                __m512d x = _mm512_maskz_loadu_pd(mask, &pop[i * D + j]);
                __m512d x_squared = _mm512_mul_pd(x, x);
                __m512d cos_term = _mm512_cos_pd(_mm512_mul_pd(vec_two_pi, x));
                __m512d result = _mm512_sub_pd(x_squared, _mm512_mul_pd(vec_ten, cos_term));
#pragma omp atomic
                fitness[i] += _mm512_mask_reduce_add_pd(mask, result);
            }
        }
    }

    int D;             // Dimension of problems
    int N;             // Population size
    int it;            // Max iteration
    vector<double> Ub; // Upper bound
    vector<double> Lb; // Lower bound
    double A;          // Strength
    double B;          // Attractiveness constant
    double G;          // Absorption coefficient
};

int main() {
    int dimen, population, max_iter;

    auto start_time = chrono::high_resolution_clock::now();

    random_device rd;
    mt19937 gen(0); // rd()
    uniform_real_distribution<> dis(-1024, 1024);

    // FA fa(256, 32, 5);
    FA fa(1024, 1024, 3);
    N = fa.N, D = fa.D;
    pop.resize(N * D);
    vector<double> best_list, best_para(D);
    vector<vector<double>> best_para_list;

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int num_cpus = CPU_COUNT(&cpu_set);
    num_threads = num_cpus;

    // Initialize population
    for (int i = 0; i < fa.N; i++) {
        for (int j = 0; j < fa.D; j++) {
            pop[i * fa.D + j] = dis(gen); // Linear indexing
        }
    }

    fa.fun2();

    auto min_iter = min_element(fitness.begin(), fitness.end());
    best_list.push_back(*min_iter);
    int arr = distance(fitness.begin(), min_iter);

    // Extract the best parameters
    best_para.resize(D);
    for (int j = 0; j < fa.D; j++) {
        best_para[j] = pop[arr * fa.D + j];
    }
    best_para_list.push_back(best_para);

    double best_iter;
    double best_ = numeric_limits<double>::max();
    vector<double> best_para_(fa.D);

    int it = 1;
    while (it < fa.it) {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < fa.D; j++) {
                double steps = fa.A * (dis(gen) - 0.5) * abs(fa.Ub[0] - fa.Lb[0]);
                double r_distance = 0;

                // #pragma omp parallel for
                for (int k = 0; k < fa.N; k++) {
                    if (fitness[i] > fitness[k]) {
                        r_distance += pow(pop[i * fa.D + j] - pop[k * fa.D + j], 2);
                        double Beta = fa.B * exp(-fa.G * r_distance);
                        double xnew = pop[i * fa.D + j] + Beta * (pop[k * fa.D + j] - pop[i * fa.D + j]) + steps;

                        xnew = min(max(xnew, fa.Lb[0]), fa.Ub[0]);
                        pop[i * fa.D + j] = xnew;

                        // Update fitness after position update
                        fa.fun2();
                        auto best_iter = min_element(fitness.begin(), fitness.end());
                        best_ = *best_iter;
                        int arr_ = distance(fitness.begin(), best_iter);

                        for (int j = 0; j < fa.D; j++) {
                            best_para_[j] = pop[arr_ * fa.D + j];
                        }
                    }
                }
            }
        }
        best_list.push_back(best_);
        best_para_list.push_back(best_para_);
        it++;
        // cout << "Iteration " << it << " finished" << endl;
    }

    // Save results to file
    ofstream file("results_1D_omp.csv");
    if (file.is_open()) {
        // Write header
        file << "Dimension_1";
        for (int d = 1; d < fa.D; ++d) {
            file << ",Dimension_" << d + 1;
        }
        file << ",Fitness\n";

        // Write population matrix and fitness
        for (int i = 0; i < fa.N; ++i) {
            for (int j = 0; j < fa.D; ++j) {
                file << pop[i * fa.D + j];
                if (j < fa.D - 1) {
                    file << ",";
                }
            }
            file << "," << fitness[i] << "\n";
        }

        // Write best fitness values
        file << "\nGeneration,Best Fitness\n";
        for (int i = 0; i < best_list.size(); ++i) {
            file << i << "," << best_list[i] << "\n";
        }
        file.close();
        // cout << "Results saved to results_1D_omp" << endl;
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_time = end_time - start_time;
    // cout << "Program execution time: " << elapsed_time.count() << " seconds" << endl;
    cout << elapsed_time.count() << endl;

    return 0;
}