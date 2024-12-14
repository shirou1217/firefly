#! /bin/bash
rm omp
icpc -O3 -qopenmp -march=native ./firefly_1D_omp.cpp -o omp
# g++ -O3 -fopenmp ./firefly_1D_omp.cpp -o omp
if [ $? -eq 0 ]; then
    echo -e "\e[92mCompile Succeed.\e[0m"
else
    echo -e "\e[91mCompile Failed.\e[0m"
    exit
fi

srun -n1 -c52 --partition=final ./omp
diff results_1D_omp.csv results_cudaV3.csv 
if [ $? -eq 0 ]; then
    echo -e "\e[92mSucceed.\e[0m"
else
    echo -e "\e[91mFailed.\e[0m"
fi