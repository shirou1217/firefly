#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <fstream>
#include <chrono>
// #include "nvtx3/nvtx3.hpp"


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
    
    vector<double> fun(const vector<vector<double>>& pop) {
        //nvtxRangePushA("fun() calculate fitness");
        vector<double> result;
        for (int i = 0; i < pop.size(); i++) {
            double funsum = 0;
            for (int j = 0; j < D; j++) {
                double x = pop[i][j];
                funsum += x * x - 10 * cos(2 * M_PI * x);
            }
            funsum += 10 * D;
            result.push_back(funsum);
        }
        //nvtxRangePop();
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
    mt19937 gen(0); //rd()
    uniform_real_distribution<> dis(-1024, 1024);
    uniform_real_distribution<> step_dis(0, 1);

    FA fa(1024, 5, 5);
    vector<vector<double>> pop(fa.N, vector<double>(fa.D));
    
   
    //nvtxRangePushA("pop initialize");
    for (int i = 0; i < fa.N; i++) {
        for (int j = 0; j < fa.D; j++) {
            pop[i][j] = dis(gen);
        }
    }
    //nvtxRangePop();


    vector<double> fitness = fa.fun(pop);

    vector<double> best_list;
    vector<vector<double>> best_para_list;

    auto min_iter = min_element(fitness.begin(), fitness.end());
    best_list.push_back(*min_iter);
    int arr = distance(fitness.begin(), min_iter);
    best_para_list.push_back(pop[arr]);
    double best_iter;
    int best_index;
    
    double r_distance = 0;
    double best_ = std::numeric_limits<double>::max(); // Initialize to a large value
    vector<double> best_para_(fa.D); // Initialize with the correct dimension
    int it = 1;
    while (it < fa.it) {    
        for (int i = 0; i < fa.N; i++) {
            for (int j = 0; j < fa.D; j++) {
                double steps = fa.A * (dis(gen) - 0.5) * abs(fa.Ub[0] - fa.Lb[0]); //step_dis(gen)
                for (int k = 0; k < fa.N; k++) {
                   //nvtxRangePushA("update firefly position & fitness");
                    if (fitness[i] > fitness[k]) {
                        r_distance += pow(pop[i][j] - pop[k][j], 2);
                        double Beta = fa.B * exp(-fa.G * r_distance);
                        double xnew = pop[i][j] + Beta * (pop[k][j] - pop[i][j]) + steps;

                        xnew = min(max(xnew, fa.Lb[0]), fa.Ub[0]);
                        pop[i][j] = xnew;
                        // Update fitness after each iteration
                        fitness = fa.fun(pop);
                        auto best_iter = min_element(fitness.begin(), fitness.end());
                        best_ = *best_iter;
                        int arr_ = distance(fitness.begin(), best_iter);
                        best_para_ = pop[arr_];
                    }
                    //nvtxRangePop();
                }
            }
        }
        best_list.push_back(best_);
        best_para_list.push_back(best_para_);
        it++;
        cout<<"iteration"<<it<<" finished"<<"\n";

    }

    //nvtxRangePushA("write result file");
    ofstream file("results_cpp.csv");
    if (file.is_open()) {
        // Write the header
        file << "Dimension_1";
        for (int d = 1; d < fa.D; ++d) {
            file << ",Dimension_" << d + 1;
        }
        file << ",Fitness\n";

        // Write the population matrix and corresponding fitness values
        for (int i = 0; i < pop.size(); ++i) {
            for (int j = 0; j < pop[i].size(); ++j) {
                file << pop[i][j];
                if (j < pop[i].size() - 1) {
                    file << ",";
                }
            }
            file << "," << fitness[i] << "\n"; // Append fitness value after the row
        }

        // Write best fitness values per generation
        file << "\nGeneration,Best Fitness\n";
        for (int i = 0; i < best_list.size(); ++i) {
            file << i << "," << best_list[i] << "\n";
        }
        file.close();
        cout << "Results saved to results_cpp.csv" << endl;
    } 
    //nvtxRangePop();
    
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_time = end_time - start_time;
    cout << "Program execution time: " << elapsed_time.count() << " seconds" << endl;

    cout << "Best value so far saved to best_value_plot.txt" << endl;

    return 0;
}
