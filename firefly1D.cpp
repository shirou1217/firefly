#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <fstream>
#include <chrono>
// #include "/home/pp24/pp24s036/firefly/NVTX/c/include/nvtx3/nvtx3.hpp"


using namespace std;

class FA {
public:
    
    FA(int dimen, int population, int max_iter)
        : D(dimen), N(population), it(max_iter), A(0.97), B(1.0), G(0.0001) {
        //nvtxRangePushA("FA() initialize parameter");
        Ub.resize(D, 3.0);
        Lb.resize(D, 0.0);
        //nvtxRangePop();
    }
    
    vector<double> fun(const vector<double>& pop) {
        vector<double> result(N); // Fitness results for each individual in the population

        for (int i = 0; i < N; i++) {
            double funsum = 0.0;
            for (int j = 0; j < D; j++) {
                double x = pop[i * D + j]; // Access the element using linear indexing
                funsum += x * x - 10 * cos(2 * M_PI * x);
            }
            funsum += 10 * D; // Add constant term
            result[i] = funsum; // Store the fitness value
            //printf("Population[%d]: Fitness = %f\n", i, funsum);
        }

        return result;
    }



    int D;                  // Dimension of problems
    int N;                  // Population size
    int it;                 // Max iteration
    vector<double> Ub;      // Upper bound
    vector<double> Lb;      // Lower bound
    double A;               // Strength
    double B;               // Attractiveness constant
    double G;               // Absorption coefficient
};


int main() {
    int dimen, population, max_iter;

    auto start_time = chrono::high_resolution_clock::now();

    random_device rd;
    mt19937 gen(0); // rd()
    uniform_real_distribution<> dis(-1024, 1024);

    FA fa(1024, 32, 5);
    vector<double> pop(fa.N * fa.D); // 1D array for population
    
    // Initialize population
    for (int i = 0; i < fa.N; i++) {
        for (int j = 0; j < fa.D; j++) {
            pop[i * fa.D + j] = dis(gen); // Linear indexing
        }
    }

    vector<double> fitness = fa.fun(pop);

    vector<double> best_list;
    vector<vector<double>> best_para_list;

    auto min_iter = min_element(fitness.begin(), fitness.end());
    best_list.push_back(*min_iter);
    int arr = distance(fitness.begin(), min_iter);

    // Extract the best parameters
    vector<double> best_para(fa.D);
    for (int j = 0; j < fa.D; j++) {
        best_para[j] = pop[arr * fa.D + j];
    }
    best_para_list.push_back(best_para);

    double best_iter;
    double best_ = numeric_limits<double>::max();
    vector<double> best_para_(fa.D);

    int it = 1;
    while (it < fa.it) {
        for (int i = 0; i < fa.N; i++) {
            for (int j = 0; j < fa.D; j++) {
                double steps = fa.A * (dis(gen) - 0.5) * abs(fa.Ub[0] - fa.Lb[0]);
                double r_distance = 0;

                for (int k = 0; k < fa.N; k++) {
                    if (fitness[i] > fitness[k]) {
                        r_distance += pow(pop[i * fa.D + j] - pop[k * fa.D + j], 2);
                        double Beta = fa.B * exp(-fa.G * r_distance);
                        double xnew = pop[i * fa.D + j] + Beta * (pop[k * fa.D + j] - pop[i * fa.D + j]) + steps;

                        xnew = min(max(xnew, fa.Lb[0]), fa.Ub[0]);
                        pop[i * fa.D + j] = xnew;

                        // Update fitness after position update
                        fitness = fa.fun(pop);
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
        cout << "Iteration " << it << " finished" << endl;
    }

    // Save results to file
    ofstream file("results_1D.csv");
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
        cout << "Results saved to results_1D" << endl;
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_time = end_time - start_time;
    cout << "Program execution time: " << elapsed_time.count() << " seconds" << endl;

    return 0;
}
