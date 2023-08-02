#BSUB -J precondSolver
#BSUB -n 36
#BSUB -o ../reports/test-cg-ball-cent-cpu-Xeon-6140-%J.log
#BSUB -e ../tmp/%J.lsf.err
#BSUB -W 10
#BSUB -q batch
#BSUB -R "span[ptile=36]"

cd ${LS_SUBCWD}/..
source ../set-oneapi.sh
# set OMP_NUM_THREADS _and_ export! 
OMP_NUM_THREADS=$LSB_DJOB_NUMPROC 
export OMP_NUM_THREADS

./bin/main