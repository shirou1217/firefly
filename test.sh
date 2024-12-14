#! /bin/bash

rm omp
rm results_1D_omp.csv 
rm record.txt

icpc -O3 -qopenmp -march=native ./firefly_1D_omp.cpp -o omp
# g++ -O3 -pthread ./firefly_1D_pthread.cpp -o pthread
if [ $? -eq 0 ]; then
    echo -e "\e[92mCompile Succeed.\e[0m"
else
    echo -e "\e[91mCompile Failed.\e[0m"
fi

cpu_nums="1 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 84 88 92 96"
# cpu_nums="44 48 52 56"
for cpu in $cpu_nums; do
    echo "computing $cpu..."
    srun -n1 -c$cpu --partition=final ./omp >> record.txt
    diff results_1D_omp.csv results_cudaV3.csv 
    if [ $? -eq 0 ]; then
      echo -e "\e[92mSucceed.\e[0m"
    else
      echo -e "\e[91mFailed.\e[0m"
      exit
    fi
done

