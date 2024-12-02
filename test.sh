# srun -n1 -c1 ./omp >> record.txt
# diff results_1D_omp.csv results_1024.csv 
# if [ $? -eq 0 ]; then
#     echo -e "\e[92mSucceed.\e[0m"
# else
#     echo -e "\e[91mFailed.\e[0m"
# fi
# srun -n1 -c4 ./omp >> record.txt
# diff results_1D_omp.csv results_1024.csv 
# if [ $? -eq 0 ]; then
#     echo -e "\e[92mSucceed.\e[0m"
# else
#     echo -e "\e[91mFailed.\e[0m"
# fi
# srun -n1 -c8 ./omp >> record.txt
# diff results_1D_omp.csv results_1024.csv 
# if [ $? -eq 0 ]; then
#     echo -e "\e[92mSucceed.\e[0m"
# else
#     echo -e "\e[91mFailed.\e[0m"
# fi
# srun -n1 -c12 ./omp >> record.txt
# diff results_1D_omp.csv results_1024.csv 
# if [ $? -eq 0 ]; then
#     echo -e "\e[92mSucceed.\e[0m"
# else
#     echo -e "\e[91mFailed.\e[0m"
# fi
srun -N2 -n1 -c16 ./omp >> record.txt
diff results_1D_omp.csv results_1024.csv 
if [ $? -eq 0 ]; then
    echo -e "\e[92mSucceed.\e[0m"
else
    echo -e "\e[91mFailed.\e[0m"
fi
srun -n1 -c20 ./omp >> record.txt
diff results_1D_omp.csv results_1024.csv 
if [ $? -eq 0 ]; then
    echo -e "\e[92mSucceed.\e[0m"
else
    echo -e "\e[91mFailed.\e[0m"
fi
srun -n1 -c24 ./omp >> record.txt
diff results_1D_omp.csv results_1024.csv 
if [ $? -eq 0 ]; then
    echo -e "\e[92mSucceed.\e[0m"
else
    echo -e "\e[91mFailed.\e[0m"
fi