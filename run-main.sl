#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J HydraGNN
#SBATCH -o SC25-multibranch-%j.out
#SBATCH -e SC25-multibranch-%j.out
#SBATCH -t 01:30:00
#SBATCH -p batch
#SBATCH -C nvme
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
##SBATCH -q debug
#SBATCH -N 5
##SBATCH -S 1

echo "====== JOB SCRIPT ======"
cat "$0"
echo "====== END OF JOB SCRIPT ======"
 
# Load conda environemnt
source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm624.sh
source /lustre/orion/lrn070/world-shared/mlupopa/max_conda_envs_frontier/bin/activate
conda activate hydragnn_rocm624
 
#export python path to use ADIOS2 v.2.9.2
export PYTHONPATH=/lustre/orion/lrn070/world-shared/mlupopa/ADIOS_ROCm624/adios2-install/lib/python3.11/site-packages/:$PYTHONPATH

which python
python -c "import numpy; print(numpy.__version__)"


echo $LD_LIBRARY_PATH  | tr ':' '\n'

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=0

export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1

export HYDRAGNN_VALTEST=0

## Checking
env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA


# export datadir0=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/ANI1x-v3.bp
# export datadir1=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/qm7x-v3.bp
# export datadir2=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/MPTrj-v3.bp
# export datadir3=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/Alexandria-v3.bp
# export datadir4=/lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/transition1x-v3.bp


set -x
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_JOB_NUM_NODES rm -rf /mnt/bb/kmehta/*
time srun -N $SLURM_JOB_NUM_NODES -n $SLURM_JOB_NUM_NODES cp -r /lustre/orion/world-shared/lrn070/HydraGNN-sc25-comm/*-v3.bp /mnt/bb/kmehta/.
set +x
if [ $? -ne 0 ]; then
    echo "Could not load data into memory. Exiting"
    exit 1
fi

export datadir0=/mnt/bb/kmehta/ANI1x-v3.bp
export datadir1=/mnt/bb/kmehta/qm7x-v3.bp
export datadir2=/mnt/bb/kmehta/MPTrj-v3.bp
export datadir3=/mnt/bb/kmehta/Alexandria-v3.bp
export datadir4=/mnt/bb/kmehta/transition1x-v3.bp


echo "Launching HydraGNN for baseline tests at `date`"
for batch_size in 32; do
    for count in 1 2 3 4 5; do
        export NTOTPROCS=$((SLURM_JOB_NUM_NODES*8))
        export HYDRAGNN_MAX_NUM_BATCH=50
        export BATCH_SIZE=$batch_size
        export EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE* NTOTPROCS))
        export LOGDIRNAME="SC25_multibranch_weakscaling_JOB${SLURM_JOB_ID}_N${SLURM_JOB_NUM_NODES}_NPROCS${NTOTPROCS}_EBS${EFFECTIVE_BATCH_SIZE}_LBS${BATCH_SIZE}_MaxNumBatch${HYDRAGNN_MAX_NUM_BATCH}_TP0_BASELINE_${count}"
        GPSUMMARY=/lustre/orion/lrn070/world-shared/kmehta/hydragnn-sc25/HydraGNN-pzhang/logs/$LOGDIRNAME
        
        if [ "$BATCH_SIZE" -lt 16 ] || [ "$BATCH_SIZE" -gt 2048 ]; then
            continue
        fi
        echo "Node count: $SLURM_JOB_NUM_NODES, num_processes: $NTOTPROCS, effective batch size: $EFFECTIVE_BATCH_SIZE, local batch size: $BATCH_SIZE"
        # this test was already performed
        if [ -e $GPSUMMARY ]; then
            echo "Skipping $GPSUMMARY as it is already done."
            continue
        fi

        echo "Node count: $SLURM_JOB_NUM_NODES, num_processes: $NTOTPROCS, effective batch size: $EFFECTIVE_BATCH_SIZE, local batch size: $BATCH_SIZE"
        echo "Experiment started at `date`"

        set -x

        timeout --signal=TERM --kill-after=10s 10m srun -n$((SLURM_JOB_NUM_NODES*8)) --ntasks-per-node=8 -c4 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/multibranch/train.py --log=$LOGDIRNAME \
        --inputfile=multibranch_GFM260_SC25.json --multi --ddstore --multi_model_list=$datadir0,$datadir1,$datadir2,$datadir3,$datadir4 --num_samples=$((EFFECTIVE_BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
        --everyone --batch_size=${BATCH_SIZE} --num_epoch=5

        set +x
        echo "Experiment ended at `date`"
    done
done
echo "Baseline tests with HydraGNN complete at `date`"


echo "Launching gaussian calculations at `date`"
bash /lustre/orion/lrn070/world-shared/kmehta/pubchem-interference-test/pubchem-gaussian/src/gaussian/run.sl &
GAUSSIAN_PID=$!

sleep 600

echo "Launching HydraGNN for interference tests at `date`"
for batch_size in 32; do
    for count in 1 2 3 4 5; do
        export NTOTPROCS=$((SLURM_JOB_NUM_NODES*8))
        export HYDRAGNN_MAX_NUM_BATCH=50
        export BATCH_SIZE=$batch_size
        export EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE* NTOTPROCS))
        export LOGDIRNAME="SC25_multibranch_weakscaling_JOB${SLURM_JOB_ID}_N${SLURM_JOB_NUM_NODES}_NPROCS${NTOTPROCS}_EBS${EFFECTIVE_BATCH_SIZE}_LBS${BATCH_SIZE}_MaxNumBatch${HYDRAGNN_MAX_NUM_BATCH}_TP0_INTERFERENCE_${count}"
        GPSUMMARY=/lustre/orion/lrn070/world-shared/kmehta/hydragnn-sc25/HydraGNN-pzhang/logs/$LOGDIRNAME
        
        if [ "$BATCH_SIZE" -lt 16 ] || [ "$BATCH_SIZE" -gt 2048 ]; then
            continue
        fi
        echo "Node count: $SLURM_JOB_NUM_NODES, num_processes: $NTOTPROCS, effective batch size: $EFFECTIVE_BATCH_SIZE, local batch size: $BATCH_SIZE"
        # this test was already performed
        if [ -e $GPSUMMARY ]; then
            echo "Skipping $GPSUMMARY as it is already done."
            continue
        fi

        echo "Node count: $SLURM_JOB_NUM_NODES, num_processes: $NTOTPROCS, effective batch size: $EFFECTIVE_BATCH_SIZE, local batch size: $BATCH_SIZE"
        echo "Experiment started at `date`"

        set -x

        timeout --signal=TERM --kill-after=10s 10m srun --exclusive -n$((SLURM_JOB_NUM_NODES*8)) --ntasks-per-node=8 -c4 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/multibranch/train.py --log=$LOGDIRNAME \
        --inputfile=multibranch_GFM260_SC25.json --multi --ddstore --multi_model_list=$datadir0,$datadir1,$datadir2,$datadir3,$datadir4 --num_samples=$((EFFECTIVE_BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
        --everyone --batch_size=${BATCH_SIZE} --num_epoch=5 

        set +x
        echo "Experiment ended at `date`"
    done
done
echo "Interference tests with HydraGNN complete at `date`"

echo "Killing gaussian"
kill $GAUSSIAN_PID

echo "ALL DONE"

