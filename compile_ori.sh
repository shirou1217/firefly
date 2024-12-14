#! /bin/bash
# g++ -O3 ./firefly1D.cpp -o ori
icpc -O3 ./firefly1D.cpp -o ori
srun --partition=final ./ori
rm ori