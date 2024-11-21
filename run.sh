g++ -O3 -fopenmp ./firefly.cpp -o firefly
srun -n1 -c32 ./firefly
diff best_value_plot.txt best_value_plot_test.txt