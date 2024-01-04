#!/bin/bash

#SBATCH --partition=rome
#SBATCH --ntasks-per-node=20
#SBATCH --nodes=2
#SBATCH -o out.log
#SBATCH -e error.log
#SBATCH --time=05:00

module load gcc openmpi curl

make openmpi

srun ./test-openmpi -l "https://dumps.wikimedia.org/other/static_html_dumps/current/el/wikipedia-el-html.tar.7z" 28084358
