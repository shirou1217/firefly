#!/bin/bash
OMPI_CXX=icpc mpicxx -O3 -march=native ./firefly1D_mpi.cpp -o mpi
if [ $? -eq 0 ]; then
    echo -e "\e[92mCompile Succeed.\e[0m"
else
    echo -e "\e[91mCompile Failed.\e[0m"
    exit
fi


srun -n24 --partition=final ./mpi
# diff results_1D_mpi.csv results_cudaV3.csv 
# if [ $? -eq 0 ]; then
#     echo -e "\e[92mSucceed.\e[0m"
# else
#     echo -e "\e[91mFailed.\e[0m"
# fi