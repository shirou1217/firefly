srun -n1 -c1 ./omp >> record.txt
echo 'hi'
srun -n1 -c2 ./omp >> record.txt
echo 'hi'
srun -n1 -c4 ./omp >> record.txt
echo 'hi'
srun -n1 -c6 ./omp >> record.txt
echo 'hi'
srun -n1 -c8 ./omp >> record.txt
echo 'hi'
srun -n1 -c10 ./omp >> record.txt
echo 'hi'
srun -n1 -c12 ./omp >> record.txt
echo 'hi'