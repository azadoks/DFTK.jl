#!/bin/bash
cd "$(dirname "$0")"
PREFIX="iron_PBE"

rm -rf ${PREFIX}.out ${PREFIX}.save/ CRASH __ABI_MPIABORTFILE__
mpirun -np 4 pw.x -in ${PREFIX}.in &> ${PREFIX}.out

FILES=(
    ${PREFIX}.save/
    CRASH
    __ABI_MPIABORTFILE__
)
rm -rf ${FILES[@]}
