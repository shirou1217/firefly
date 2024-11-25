#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
//#include "/nvtx3/nvtx3.hpp"


using namespace std;

// CUDA version of the fitness function (fun)

// double* arr[1024];
// 1 block, 1024 threads: 1024 elements
// warp_reduce: google
// 1 block, 32 ~ 128 threads: 1024 elements

// 1024 threads -> 32 warp -> CUDA -> 1 warp = 32 threads
// thread in a wrap can use register to swap data -> each wrap use wrap reduce -> each wrap has only 1 sum
// write 32 sum to shared memory
// let 1 wrap to read shared memory
// do wrap reduce, write back to the fitness
// swap data between wraps -> use shared memory

// N blocks, each block 1024 threads, block i deal with pop[i][0~D-1]


// [[...],
//  [...],
//  [...]]

// pop: N x D, fitness: N


//one wrap calculate one population
// __global__ void fun_kernel(double* pop, double* fitness, int N, int D) {
//     using WarpReduce = cub::WarpReduce<double>;

//     // 分配共享記憶體，每個 warp 使用獨立的 32 個 slot
//     __shared__ double shared_funsum[32 * 32];
//     __shared__ typename WarpReduce::TempStorage temp_storage[32]; // 每個 warp 的暫存空間

//     // Warp 和 Thread 的索引
//     int warp_id = threadIdx.x / 32;   // Warp 在 block 內的索引
//     int lane_id = threadIdx.x % 32;  // Thread 在 warp 內的索引

//     // Warp 負責的 population 索引
//     int i = blockIdx.x * blockDim.x / 32 + warp_id;

//     if (i < N) { // 確保 population 索引合法
//         double funsum = 0.0;

//         // 每個 thread 負責連續的部分維度
//         int chunk_size = D / 32;       // 每個 thread 處理的維度數
//         int start_idx = lane_id * chunk_size; // 每個 thread 起始維度
//         int end_idx = start_idx + chunk_size; // 每個 thread 結束維度

//         // 計算該 thread 的部分
//         for (int j = start_idx; j < end_idx; j++) {
//             double x = pop[i * D + j];
//             funsum += x * x - 10 * cos(2 * M_PI * x);
//         }

//         // 將 thread 的計算結果存入共享記憶體
//         shared_funsum[warp_id * 32 + lane_id] = funsum;
//         __syncthreads(); // 確保所有 thread 完成共享記憶體的寫入

//         // Lane 0 使用 WarpReduce 對共享記憶體的數據進行加總
//         double warp_sum = WarpReduce(temp_storage[warp_id]).Sum(shared_funsum[warp_id * 32 + lane_id]);

//         // Lane 0 將加總結果寫回 fitness
//         if (lane_id == 0) {
//             warp_sum += 10 * D;
//             fitness[i] = warp_sum;
//             printf("warp_id = %d warp_sum = %f index = %d\n",warp_id,warp_sum,i);
//         }
//     }
// }
// 1024 block all wrap one population
__global__ void fun_kernel(double* pop, double* fitness, int N, int D) {
    using WarpReduce = cub::WarpReduce<double>;

    // 每個 Warp 的共享記憶體區域，用於存儲每個 Warp 的計算結果
    __shared__ double shared_funsum[32];
    __shared__ typename WarpReduce::TempStorage warp_temp_storage[32]; // 每個 Warp 的暫存空間

    // Warp 和 Thread 的索引
    int warp_id = threadIdx.x / 32;   // Warp 在 block 內的索引
    int lane_id = threadIdx.x % 32;  // Thread 在 warp 內的索引

    // 每個 Block 負責的 population 索引
    int i = blockIdx.x;

    if (i < N) { // 確保 population 索引合法
        double funsum = 0.0;

        // 每個 Warp 負責部分維度
        int dims_per_warp = D / 32;       // 每個 Warp 處理的維度數
        int dims_per_thread = dims_per_warp / 32; // 每個 Thread 處理的連續維度數
        int start_idx = warp_id * dims_per_warp + lane_id * dims_per_thread; // 每個 Thread 的起始維度
        int end_idx = start_idx + dims_per_thread; // 每個 Thread 的結束維度

        // 每個 Thread 處理連續的數據
        for (int j = start_idx; j < end_idx; j++) {
            double x = pop[i * D + j];
            funsum += x * x - 10 * cos(2 * M_PI * x);
        }

        // 每個 Warp 的 Lane 0 執行加總，結果存入共享記憶體
        double warp_sum = WarpReduce(warp_temp_storage[warp_id]).Sum(funsum);
        if (lane_id == 0) {
            shared_funsum[warp_id] = warp_sum;
            //printf("warp_id = %d warp_sum = %f\n",warp_id,warp_sum);
        }
        __syncthreads(); // 確保所有 Warp 的結果都存入共享記憶體

        // Warp 0 的 Lane 0 對共享記憶體中的 32 個值進行加總
        if (warp_id == 0) {
            double total_sum = (lane_id < 32) ? shared_funsum[lane_id] : 0.0;
            total_sum = WarpReduce(warp_temp_storage[0]).Sum(total_sum);

            if (lane_id == 0) {
                total_sum += 10 * D;
                fitness[i] = total_sum;
                //printf("Population[%d]: Fitness = %f\n", i, total_sum);
            }
        }
    }
}

//one block 32 warp calculate one population
// __global__ void fun_kernel(double* pop, double* fitness, int N, int D) {
//     using WarpReduce = cub::WarpReduce<double>;

//     // 每個 Warp 的共享記憶體區域，用於存儲計算結果
//     __shared__ double shared_funsum[32];
//     __shared__ typename WarpReduce::TempStorage warp_temp_storage[32]; // 每個 Warp 的暫存空間

//     // Warp 和 Thread 的索引
//     int warp_id = threadIdx.x / 32;   // Warp 在 Block 內的索引
//     int lane_id = threadIdx.x % 32;  // Thread 在 Warp 內的索引

//     for (int i = 0; i < N; i++) { // 每次計算一個 fitness[i]
//         double funsum = 0.0;

//         // 每個 Warp 負責部分維度
//         int dims_per_warp = D / 32;       // 每個 Warp 處理的維度數
//         int dims_per_thread = dims_per_warp / 32; // 每個 Thread 處理的連續維度數
//         int start_idx = warp_id * dims_per_warp + lane_id * dims_per_thread; // 每個 Thread 的起始維度
//         int end_idx = start_idx + dims_per_thread; // 每個 Thread 的結束維度

//         // 每個 Thread 處理連續的數據
//         for (int j = start_idx; j < end_idx; j++) {
//             double x = pop[i * D + j];
//             funsum += x * x - 10 * cos(2 * M_PI * x);
//         }

//         // 使用 WarpReduce 計算該 Warp 的加總
//         double warp_sum = WarpReduce(warp_temp_storage[warp_id]).Sum(funsum);
//         if (lane_id == 0) {
//             shared_funsum[warp_id] = warp_sum;
//         }
//         __syncthreads(); // 確保所有 Warp 的結果都存入共享記憶體

//         // Warp 0 的 Lane 0 對共享記憶體中的 32 個值進行加總
//         if (warp_id == 0) {
//             double total_sum = (lane_id < 32) ? shared_funsum[lane_id] : 0.0;
//             total_sum = WarpReduce(warp_temp_storage[0]).Sum(total_sum);

//             if (lane_id == 0) {
//                 total_sum += 10 * D;
//                 fitness[i] = total_sum;
//                // printf("Fitness[%d]: %f\n", i, total_sum);
//             }
//         }
//         __syncthreads(); // 確保所有 Thread 準備好進行下一個 fitness[i] 的計算
//     }
// }




class FA {
public:
    FA(int dimen, int population, int max_iter)
        : D(dimen), N(population), it(max_iter), A(0.97), B(1.0), G(0.0001) {
        Ub.resize(D, 3.0);
        Lb.resize(D, 0.0);
        // Allocate GPU memory
        nvtxRangePushA("Malloc");
        cudaMalloc(&d_pop, N * D * sizeof(double));
        cudaMalloc(&d_fitness, N * sizeof(double));
        nvtxRangePop();
    }
    
    ~FA() {
        // Free GPU memory once
        nvtxRangePushA("cudaFree");
        cudaFree(d_pop);
        cudaFree(d_fitness);
        nvtxRangePop();
    }
    vector<double> fun(const vector<double>& pop) {
        nvtxRangePushA("fun() calculate fitness");
        std::vector<double> h_fitness(N);

        // Copy data to GPU
        nvtxRangePushA("cudaMemcpyHostToDevice");
        cudaMemcpy(d_pop, pop.data(),N * D * sizeof(double), cudaMemcpyHostToDevice);
        nvtxRangePop();
        // Launch kernel
        int blockSize = 1024;
        // int numBlocks = (N + blockSize - 1) / blockSize;
        
        // int numBlocks = (N + 31) / 32;
        int numBlocks = N; 

        // GPU A100 has 108SM, each SM can compute multi-block
        // 1 block -> SM
        // std::cerr << "numBlocks: " << numBlocks << std::endl;
        fun_kernel<<<numBlocks, blockSize>>>(d_pop, d_fitness, N, D);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        }
        // Copy results back to CPU
        nvtxRangePushA("cudaMemcpyDeviceToHost");
        cudaMemcpy(h_fitness.data(), d_fitness, N * sizeof(double), cudaMemcpyDeviceToHost);
        nvtxRangePop();

        nvtxRangePop();
        return h_fitness;
    }

    int D;                  // Dimension of problems
    int N;                  // Population size
    int it;                 // Max iteration
    vector<double> Ub;      // Upper bound
    vector<double> Lb;      // Lower bound
    double A;               // Strength
    double B;               // Attractiveness constant
    double G;               // Absorption coefficient
private:
    double* d_pop;          // GPU memory for population
    double* d_fitness;      // GPU memory for fitness
};

int main() {
    int dimen, population, max_iter;

    auto start_time = chrono::high_resolution_clock::now();

    random_device rd;
    mt19937 gen(0); // rd()
    uniform_real_distribution<> dis(-1024, 1024);

    FA fa(1024, 32, 5);
    vector<double> pop(fa.N * fa.D); // 1D array for population
    
    {// Initialize population
        nvtx3::scoped_range scope{"Init pop"};
        for (int i = 0; i < fa.N; i++) {
            for (int j = 0; j < fa.D; j++) {
                pop[i * fa.D + j] = dis(gen); // Linear indexing
            }
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
         nvtx3::scoped_range scope_it{"iter = " + std::to_string(it), nvtx3::rgb{255,218,185}};
        for (int i = 0; i < fa.N; i++) {
            nvtx3::scoped_range scope_i{"i = " + std::to_string(i)};  // Range for a scope
            for (int j = 0; j < fa.D; j++) {
                double steps = fa.A * (dis(gen) - 0.5) * abs(fa.Ub[0] - fa.Lb[0]);
                double r_distance = 0;
                nvtx3::scoped_range scope_j{"j = " + std::to_string(j)};  // Range for a scope
                for (int k = 0; k < fa.N; k++) {
                    nvtx3::scoped_range scope_k{"k = " + std::to_string(k)};  // Range for a scope
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
    
    nvtxRangePushA("write result file");
    // Save results to file
    ofstream file("results_cuda.csv");
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
        cout << "Results saved to results_cuda.csv" << endl;
    }
    nvtxRangePop();

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_time = end_time - start_time;
    cout << "Program execution time: " << elapsed_time.count() << " seconds" << endl;

    return 0;
}
