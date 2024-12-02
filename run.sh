#!/bin/bash
mpicxx -O3 ./firefly1D_mpi.cpp -o mpi
srun -n12 ./mpi
diff results_1D_mpi.csv results_1D.csv 
if [ $? -eq 0 ]; then
    echo -e "\e[92mSucceed.\e[0m"
else
    echo -e "\e[91mFailed.\e[0m"
fi