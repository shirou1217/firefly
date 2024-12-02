# srun -n1 ./mpi >> record.txt
# diff results_1D_mpi.csv results_1024.csv 
# if [ $? -eq 0 ]; then
#     echo -e "\e[92mSucceed.\e[0m"
# else
#     echo -e "\e[91mFailed.\e[0m"
# fi
# srun -n4 ./mpi >> record.txt
# diff results_1D_mpi.csv results_1024.csv 
# if [ $? -eq 0 ]; then
#     echo -e "\e[92mSucceed.\e[0m"
# else
#     echo -e "\e[91mFailed.\e[0m"
# fi
# srun -n8 ./mpi >> record.txt
# diff results_1D_mpi.csv results_1024.csv 
# if [ $? -eq 0 ]; then
#     echo -e "\e[92mSucceed.\e[0m"
# else
#     echo -e "\e[91mFailed.\e[0m"
# fi
# srun -n12 ./mpi >> record.txt
# diff results_1D_mpi.csv results_1024.csv 
# if [ $? -eq 0 ]; then
#     echo -e "\e[92mSucceed.\e[0m"
# else
#     echo -e "\e[91mFailed.\e[0m"
# fi
srun -n16 ./mpi >> record.txt
diff results_1D_mpi.csv results_1024.csv 
if [ $? -eq 0 ]; then
    echo -e "\e[92mSucceed.\e[0m"
else
    echo -e "\e[91mFailed.\e[0m"
fi
srun -n20 ./mpi >> record.txt
diff results_1D_mpi.csv results_1024.csv 
if [ $? -eq 0 ]; then
    echo -e "\e[92mSucceed.\e[0m"
else
    echo -e "\e[91mFailed.\e[0m"
fi
srun -n24 ./mpi >> record.txt
diff results_1D_mpi.csv results_1024.csv 
if [ $? -eq 0 ]; then
    echo -e "\e[92mSucceed.\e[0m"
else
    echo -e "\e[91mFailed.\e[0m"
fi