#!/bin/bash
echo "export CUDA_HOME=/usr/local/cuda-12.6">> /root/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.6/lib64">> /root/.bashrc
echo "export PATH=$PATH:/usr/local/cuda-12.6/bin">>/root/.bashrc