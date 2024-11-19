#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <fstream>
#include <chrono>
#include "/home/pp24/pp24s036/firefly/NVTX/c/include/nvtx3/nvtx3.hpp"


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
        return result;
        //nvtxRangePop();
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
    cout << "Enter dimension, population, and max iterations: ";
    cin >> dimen >> population >> max_iter;

    auto start_time = chrono::high_resolution_clock::now();

    random_device rd;
    mt19937 gen(rd()); //
    uniform_real_distribution<> dis(-100, 100);

    FA fa(dimen, population, max_iter);
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

    int it = 1;
    while (it < fa.it) {
        for (int i = 0; i < fa.N; i++) {
            for (int j = 0; j < fa.D; j++) {
                double steps = fa.A * (dis(gen) / 100.0 - 0.5) * abs(fa.Ub[0] - fa.Lb[0]);
                double r_distance = 0;

                for (int k = 0; k < fa.N; k++) {
                   //nvtxRangePushA("update firefly position");
                    if (fitness[i] > fitness[k]) {
                        r_distance += pow(pop[i][j] - pop[k][j], 2);
                        double Beta = fa.B * exp(-fa.G * r_distance);
                        double xnew = pop[i][j] + Beta * (pop[k][j] - pop[i][j]) + steps;

                        xnew = min(max(xnew, fa.Lb[0]), fa.Ub[0]);
                        pop[i][j] = xnew;
                    }
                    //nvtxRangePop();

                }
            }
        }
        
        // Update fitness after each iteration
        fitness = fa.fun(pop);

        //nvtxRangePushA("update best fitness");
        auto best_iter = min_element(fitness.begin(), fitness.end());
        best_list.push_back(*best_iter);
        int best_index = distance(fitness.begin(), best_iter);
        best_para_list.push_back(pop[best_index]);
        //nvtxRangePop();


        it++;
    }

    //nvtxRangePushA("write result file");
    ofstream file("best_value_plot.txt");
    if (file.is_open()) {
        for (int i = 0; i < best_list.size(); i++) {
            file << i << " " << best_list[i] << "\n";
        }
        file.close();
    }
    //nvtxRangePop();
    
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_time = end_time - start_time;
    cout << "Program execution time: " << elapsed_time.count() << " seconds" << endl;

    cout << "Best value so far saved to best_value_plot.txt" << endl;

    return 0;
}
