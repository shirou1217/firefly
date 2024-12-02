#include <algorithm>
#include <chrono>
#include <cmath>
#include <emmintrin.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <random>
#include <vector>
// #include "/home/pp24/pp24s036/firefly/NVTX/c/include/nvtx3/nvtx3.hpp"

using namespace std;

class FA {
  public:
    FA(int dimen, int population, int max_iter, int &size)
        : D(dimen), N(population), it(max_iter), A(0.97), B(1.0), G(0.0001) {
        // nvtxRangePushA("FA() initialize parameter");
        Ub.resize(D, 3.0);
        Lb.resize(D, 0.0);

        req_send.resize(size);
        req_recv.resize(size);
        st.resize(size);
        en.resize(size);
        for (int i = 1; i < size; i++) {
            int start = (i - 1) * (N / (size - 1));
            int end = i == size - 1 ? N : start + (N / (size - 1));
            st[i] = start, en[i] = end;
        }
        // nvtxRangePop();
    }

    void fun(double *pop, double *result, int &size, int &rank) {
        if (rank == 0) {
            MPI_Request request;
            MPI_Ibcast(&rank, 1, MPI_INT, 0, MPI_COMM_WORLD, &request);
            // MPI_Bcast(&rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
            for (int i = 1; i < size; ++i) {
                int start = st[i];
                int end = en[i];
                // MPI_Send(pop + start * D, (end - start) * D, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Isend(pop + start * D, (end - start) * D, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &req_send[i - 1]);
            }
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            MPI_Waitall(size - 1, req_send.data(), MPI_STATUSES_IGNORE);

            for (int i = 1; i < size; ++i) {
                int start = st[i];
                int end = en[i];
                // MPI_Recv(result + start, end - start, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Irecv(result + start, end - start, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &req_recv[i - 1]);
            }
            MPI_Waitall(size - 1, req_recv.data(), MPI_STATUSES_IGNORE);
        } else {
            int start = st[rank];
            int end = en[rank];
            // MPI_Recv(pop + start * D, (end - start) * D, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, 0);
            MPI_Irecv(pop + start * D, (end - start) * D, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req_recv[0]);
            MPI_Wait(&req_recv[0], MPI_STATUS_IGNORE);

            for (int i = start; i < end; ++i) {
                result[i] = 10 * D;
                for (int j = 0; j < D; ++j) {
                    double x = pop[i * D + j];
                    result[i] += x * x - 10 * cos(2 * M_PI * x);
                }
            }

            MPI_Send(result + start, end - start, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
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

    vector<MPI_Request> req_send, req_recv;
    vector<int> st, en;
};

int main(int argc, char **argv) {
    auto start_time = chrono::high_resolution_clock::now();
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int dimen, population, max_iter;
    FA fa(256, 32, 5, size);
    double *pop, *fitness;
    int N = fa.N, D = fa.D;
    pop = (double *)malloc(N * D * sizeof(double));
    fitness = (double *)malloc(N * sizeof(double));

    if (rank != 0) {
        int flag = 0;
        while (1) {
            // MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Request request;
            MPI_Ibcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            if (flag == -1)
                break;
            fa.fun(pop, fitness, size, rank);
        }
        // cout << "rank" << rank << " finish!\n";
    } else {

        random_device rd;
        mt19937 gen(0); // rd()
        uniform_real_distribution<> dis(-1024, 1024);

        // Initialize population
        for (int i = 0; i < fa.N; i++) {
            for (int j = 0; j < fa.D; j++) {
                pop[i * fa.D + j] = dis(gen); // Linear indexing
            }
        }

        // fa.fun(pop, fitness);
        fa.fun(pop, fitness, size, rank);

        vector<double> best_list;
        vector<vector<double>> best_para_list;

        int min_iter = min_element(fitness, fitness + N) - fitness;
        best_list.push_back(fitness[min_iter]);
        int arr = distance(fitness, fitness + min_iter);

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

                    // #pragma omp parallel for
                    for (int k = 0; k < fa.N; k++) {
                        if (fitness[i] > fitness[k]) {
                            r_distance += pow(pop[i * fa.D + j] - pop[k * fa.D + j], 2);
                            double Beta = fa.B * exp(-fa.G * r_distance);
                            double xnew = pop[i * fa.D + j] + Beta * (pop[k * fa.D + j] - pop[i * fa.D + j]) + steps;

                            xnew = min(max(xnew, fa.Lb[0]), fa.Ub[0]);
                            pop[i * fa.D + j] = xnew;

                            // Update fitness after position update
                            fa.fun(pop, fitness, size, rank);

                            int best_iter = min_element(fitness, fitness + N) - fitness;
                            best_ = fitness[best_iter];
                            int arr_ = distance(fitness, fitness + best_iter);

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
        int tmp = -1;
        // MPI_Bcast(&tmp, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Request request;
        MPI_Ibcast(&tmp, 1, MPI_INT, 0, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);

        // Save results to file
        string file_name = "results_1D_mpi.csv";
        ofstream file(file_name);
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
            cout << "Results saved to " << file_name << endl;
        }
    }
    MPI_Finalize();
    if (rank == 0) {
        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed_time = end_time - start_time;
        cout << "Program execution time: " << elapsed_time.count() << " seconds" << endl;
    }

    return 0;
}