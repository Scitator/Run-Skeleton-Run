#!/usr/bin/env bash
conda upgrade pip -y && \
	conda install -c conda-forge lapack git -y && \
	conda install ipython libgcc -y && \
	conda install pytorch torchvision -c soumith -y && \
	pip install tensorflow==1.3.0 gym mpi4py && \
	pip install git+https://github.com/stanfordnmbl/osim-rl.git