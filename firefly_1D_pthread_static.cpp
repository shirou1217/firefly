#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <emmintrin.h>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <pthread.h>
#include <random>
#include <vector>
// #include "/home/pp24/pp24s036/firefly/NVTX/c/include/nvtx3/nvtx3.hpp"

using namespace std;

int N, D;
vector<double> fitness, pop;
bool done;
int cur_i;
volatile int finishes;
int num_threads;
pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_var = PTHREAD_COND_INITIALIZER;
pthread_cond_t main_cond_var = PTHREAD_COND_INITIALIZER;
pthread_barrier_t barrier;
const double ten = 10.0;
const double two_pi = 2.0 * M_PI;
__m512d vec_ten = _mm512_set1_pd(ten);       // Broadcast 10.0
__m512d vec_two_pi = _mm512_set1_pd(two_pi); // Broadcast 2 * PI
void fun3(int start, int end) {
    // cout << "fun3 " << start << " " << end << endl;

    for (int i = start; i < end; i++) {
        fitness[i] = 10 * D;

        int j = 0;
        for (; j + 7 < D; j += 8) {
            __m512d x = _mm512_loadu_pd(&pop[i * D + j]);
            __m512d x_squared = _mm512_mul_pd(x, x);
            __m512d cos_term = _mm512_cos_pd(_mm512_mul_pd(vec_two_pi, x));
            __m512d result = _mm512_sub_pd(x_squared, _mm512_mul_pd(vec_ten, cos_term));
            fitness[i] += _mm512_reduce_add_pd(result);
        }

        // remaining
        for (; j < D; j++) {
            double x = pop[i * D + j];
            fitness[i] += x * x - 10 * cos(2 * M_PI * x);
        }
    }
}

int per_job = 8;
bool start_fun, finish_fun;
int sz, rem;
void *find_job(void *args) {
    int t = *(int *)args;
    int st, en;
    if (t < rem) {
        st = t * (sz + 1);
        en = st + sz + 1;
    } else {
        st = t * sz + rem;
        en = st + sz;
    }
    // cout << t << ": " << st << ' ' << en << endl;
    int id;

    // wait to start to work
    while (1) {
        pthread_barrier_wait(&barrier);
        if (done)
            break;
        fun3(st, en);
        pthread_mutex_lock(&mutex1);
        // cout << t << "end computing\n";
        finishes += en - st;
        pthread_cond_signal(&main_cond_var);
        pthread_mutex_unlock(&mutex1);

        /*
        pthread_mutex_lock(&mutex1);
        while (!start_fun && !done) {
            cout << t << " is waiting\n";
            pthread_cond_wait(&cond_var, &mutex1);
        }
        if (done) {
            pthread_mutex_unlock(&mutex1);
            break;
        }
        cout << t << " start computing\n";
        pthread_mutex_unlock(&mutex1);

        fun3(st, en);
        pthread_mutex_lock(&mutex1);
        cout << t << "end computing\n";
        finishes += en - st;
        pthread_cond_signal(&main_cond_var);
        pthread_mutex_unlock(&mutex1);
        */
    }

    return NULL;
}

class FA {
  public:
    FA(int dimen, int population, int max_iter)
        : D(dimen), N(population), it(max_iter), A(0.97), B(1.0), G(0.0001) {
        // nvtxRangePushA("FA() initialize parameter");
        Ub.resize(D, 3.0);
        Lb.resize(D, 0.0);
        // nvtxRangePop();
    }

    // void fun(const vector<double> &pop, vector<double> &result) {
    void fun() {
        pthread_barrier_wait(&barrier);

        pthread_mutex_lock(&mutex1);
        while (finishes < N) {
            pthread_cond_wait(&main_cond_var, &mutex1);
        }
        start_fun = 0;
        finishes = 0;
        pthread_mutex_unlock(&mutex1);
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

    // FA fa(32, 32, 5);
    FA fa(1024, 1024, 3);
    N = fa.N, D = fa.D;
    pop.resize(fa.N * fa.D); // 1D array for population
    fitness.resize(fa.N);

    // Initialize population
    for (int i = 0; i < fa.N; i++) {
        for (int j = 0; j < fa.D; j++) {
            pop[i * fa.D + j] = dis(gen); // Linear indexing
        }
    }

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int num_cpus = CPU_COUNT(&cpu_set);
    num_threads = min(num_cpus, N);
    sz = N / num_threads, rem = N % num_threads;
    // per_job = 8;
    per_job = N / num_threads;
    // cout << "cpu nums: " << num_cpus << endl;
    pthread_t threads[num_threads];
    vector<int> a(num_threads);
    cur_i = finishes = 0;
    done = 0;
    start_fun = 0;
    pthread_barrier_init(&barrier, NULL, num_threads + 1);
    for (int i = 0; i < num_threads; i++) {
        a[i] = i;
        pthread_create(&threads[i], NULL, find_job, &a[i]);
    }
    fa.fun();

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
        for (int i = 0; i < 10; i++) {
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
                        // fitness = fa.fun(pop);
                        fa.fun();
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
    cout << "all done\n";
    done = 1;
    pthread_barrier_wait(&barrier);
    // pthread_mutex_lock(&mutex1);
    // done = true;
    // pthread_cond_broadcast(&cond_var);
    // pthread_mutex_unlock(&mutex1);
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Save results to file
    ofstream file("results_1D_pthread.csv");
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
        // cout << "Results saved to results_1D_pthread" << endl;
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_time = end_time - start_time;
    cout << num_cpus << ", " << elapsed_time.count() << endl;
    // cout << elapsed_time.count() << endl;

    return 0;
}