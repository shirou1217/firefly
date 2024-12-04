icpc -O3 -qopenmp -ftree-vectorize ./firefly_1D_omp.cpp -o omp
# g++ -O3 -fopenmp -ftree-vectorize ./firefly_1D.cpp -o 1D
srun -n1 -c12 ./omp
# echo 'finish'
diff results_1D_omp.csv results_cudaV3.csv 
if [ $? -eq 0 ]; then
    echo -e "\e[92mSucceed.\e[0m"
else
    echo -e "\e[91mFailed.\e[0m"
fi