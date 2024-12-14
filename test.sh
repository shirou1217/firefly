#! /bin/bash

rm pthread 
rm results_1D_pthread.csv 
rm record.txt

icpc -O3 -pthread -march=native ./firefly_1D_pthread.cpp -o pthread
# g++ -O3 -pthread ./firefly_1D_pthread.cpp -o pthread
if [ $? -eq 0 ]; then
    echo -e "\e[92mCompile Succeed.\e[0m"
else
    echo -e "\e[91mCompile Failed.\e[0m"
fi

cpu_nums="1 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 84 88 92 96"
for cpu in $cpu_nums; do
    echo "computing $cpu..."
    srun -n1 -c$cpu ./pthread >> record.txt 2> err
    diff results_1D_pthread.csv results_cudaV3.csv 
    if [ $? -eq 0 ]; then
      echo -e "\e[92mSucceed.\e[0m"
    else
      echo -e "\e[91mFailed.\e[0m"
      exit
    fi
done


# srun -n1 -c20 ./omp >> record.txt
# diff results_1D_pthread.csv results_cudaV3.csv 
# if [ $? -eq 0 ]; then
#     echo -e "\e[92mSucceed.\e[0m"
# else
#     echo -e "\e[91mFailed.\e[0m"
# fi
# srun -n1 -c24 ./omp >> record.txt
# diff results_1D_pthread.csv results_cudaV3.csv 
# if [ $? -eq 0 ]; then
#     echo -e "\e[92mSucceed.\e[0m"
# else
#     echo -e "\e[91mFailed.\e[0m"
# fi