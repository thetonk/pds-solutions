#!/bin/bash

#SBATCH --job-name=isingTest
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=results.log
#SBATCH --error=error.log
#SBATCH --time=10:00

module load gcc/10.2.0 cuda/11.1.0
nvidia-smi
make all

dimension="$1"
iterations="$2"

echo "testing sequential!"
./test-ising "$dimension" "$iterations" > out.log
echo "testing CUDA V1!"
./test-isingV1 "$dimension" "$iterations" > outV1.log
echo "testing CUDA V2!"
./test-isingV2 "$dimension" "$iterations" > outV2.log
echo "testing CUDA V3!"
./test-isingV3 "$dimension" "$iterations" > outV3.log

latticeSeq="$(head -n -1 out.log)"
latticeV1="$(head -n -1 outV1.log)"
latticeV2="$(head -n -1 outV2.log)"
latticeV3="$(head -n -1 outV3.log)"

# in order for this to work,you must enable printing the resulting lattices
if [[ "$latticeSeq" == "$latticeV1" && "$latticeV2"  == "$latticeSeq" && "$latticeV3" == "$latticeSeq" ]]; then
    echo "OK! All implentations produce the same lattice!"
else
    echo "Warning! Lattices vary when compared to each other!"
fi

echo "RESULT TIMES"
echo "Sequential; $(tail -1 out.log)"
echo "CUDA V1; $(tail -1 outV1.log)"
echo "CUDA V2; $(tail -1 outV2.log)"
echo "CUDA V3; $(tail -1 outV3.log)"

rm out*.log
