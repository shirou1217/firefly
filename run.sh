#! /bin/bash
icpc -O3 -pthread -march=native ./firefly_1D_pthread.cpp -o pthread
# g++ -O3 -pthread ./firefly_1D_pthread.cpp -o pthread
if [ $? -eq 0 ]; then
    echo -e "\e[92mCompile Succeed.\e[0m"
else
    echo -e "\e[91mCompile Failed.\e[0m"
fi

srun -n1 -c96 ./pthread 2> err
# echo 'finish'
# diff results_1D_pthread.csv results_1D.csv 
diff results_1D_pthread.csv results_cudaV3.csv 
if [ $? -eq 0 ]; then
    echo -e "\e[92mSucceed.\e[0m"
else
    echo -e "\e[91mFailed.\e[0m"
fi

# rm results_1D_pthread.csv 
rm pthread