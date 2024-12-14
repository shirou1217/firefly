#! /bin/bash

dir_name='nsys_reports_16ranks'

mkdir -p $dir_name

# Output to ./nsys_reports/rank_$N.nsys-rep
nsys profile \
    -o "./$dir_name/rank_$PMI_RANK.nsys-rep" \
    --mpi-impl openmpi \
    --trace mpi,ucx,osrt,nvtx \
    $@
