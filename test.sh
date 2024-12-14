#! /bin/bash

rm mpi
rm results_1D_mpi.csv 
rm record.txt

OMPI_CXX=icpc mpicxx -O3 -march=native ./firefly1D_mpi.cpp -o mpi
# g++ -O3 -pthread ./firefly_1D_pthread.cpp -o pthread
if [ $? -eq 0 ]; then
    echo -e "\e[92mCompile Succeed.\e[0m"
else
    echo -e "\e[91mCompile Failed.\e[0m"
fi

rank_nums="1 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 84 88 92 96"
# cpu_nums="44 48 52 56"
for rank in $rank_nums; do
    echo "computing $rank..."
    srun -n$rank --partition=final ./mpi >> record.txt
    diff results_1D_mpi.csv results_cudaV3.csv 
    if [ $? -eq 0 ]; then
      echo -e "\e[92mSucceed.\e[0m"
    else
      echo -e "\e[91mFailed.\e[0m"
      exit
    fi
done
