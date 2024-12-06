icpc -O3 -qopenmp ./firefly_1D_omp.cpp -o omp
# g++ -O3 -fopenmp ./firefly_1D_omp.cpp -o omp
srun -n1 -c96 ./omp
# diff results_1D_omp.csv results_cudaV3.csv 
# if [ $? -eq 0 ]; then
#     echo -e "\e[92mSucceed.\e[0m"
# else
#     echo -e "\e[91mFailed.\e[0m"
# fi