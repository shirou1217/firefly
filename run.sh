icpc -O3 -qopenmp -ftree-vectorize ./firefly_1D_omp.cpp -o 1D
# g++ -O3 -fopenmp -ftree-vectorize ./firefly_1D.cpp -o 1D
srun -n1 -c12 ./1D