#!/bin/bash
export PYTHONPATH="/home/abhishek/.local/lib/python3.8/site-packages:$PYTHONPATH"
/usr/local/cuda-11.4/bin/nvprof \
  --metrics dram_read_throughput,dram_write_throughput,dram_read_transactions,dram_write_transactions,l2_read_hit_rate,l2_write_hit_rate,achieved_occupancy,sm_efficiency,warp_execution_efficiency,stall_memory_dependency,stall_not_selected,stall_exec_dependency \
  python3 /data/frodo/abhishek/GPU_STUDY/gemv_profile.py


# /usr/local/cuda-11.4/bin/nvprof \
# --print-gpu-trace  python3 /data/frodo/abhishek/GPU_STUDY/gemv_profile.py