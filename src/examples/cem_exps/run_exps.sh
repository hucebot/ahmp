#!/bin/bash

# Save current directory path
cwd=$(pwd)

mkdir -p results/
cd results/

# Handrail #
# echo "TALOS: Handrail"
#
# mkdir -p handrails/
# cd handrails
#
# for i in $(seq 1 10); do
#     mkdir -p "run_$i"/
#     cd "run_$i"/
#     rm -rf *
#
#     python $cwd/trajopt_parallel.py --exp handrails --robot talos
#
#     cd ../
# done
# cd ../ # cd to results/

###########
# Chimney #
###########

mkdir -p chimney/
cd chimney

# Climb Low #
echo "TALOS: Climb low"

mkdir -p climb_low/
cd climb_low

for i in $(seq 1 10); do
    mkdir -p "run_$i"/
    cd "run_$i"/
    rm -rf *

    python $cwd/trajopt_parallel.py --exp chimney --robot talos --dz 1.0

    cd ../
done
cd ../ # cd to chimney/

# Climb High #
# echo "TALOS: Climb high"
#
# mkdir -p climb_high/
# cd climb_high
#
# for i in $(seq 1 10); do
#     mkdir -p "run_$i"/
#     cd "run_$i"/
#     rm -rf *
#
#     python $cwd/trajopt_parallel.py --exp chimney --robot talos --dz 3.0
#
#     cd ../
# done
# cd ../ # cd to chimney/
#
# cd ../ # cd to results/

# mkdir -p ablation_study/
# cd ablation_study

# # Ablation 30% #
# echo "TALOS: Ablation 30%"
#
# mkdir -p ablate_30/
# cd ablate_30
#
# for i in $(seq 1 10); do
#     mkdir -p "run_$i"/
#     cd "run_$i"/
#     rm -rf *
#
#     python $cwd/trajopt_parallel.py --exp chimney --robot talos --dz 3.0 --abl 0.3
#
#     cd ../
# done
# cd ../ # cd to ablation_study/

# # Ablation 50% #
# echo "TALOS: Ablate 50%"
#
# mkdir -p ablate_50/
# cd ablate_50
#
# for i in $(seq 1 10); do
#     mkdir -p "run_$i"/
#     cd "run_$i"/
#     rm -rf *
#
#     python $cwd/trajopt_parallel.py --exp chimney --robot talos --dz 3.0 --abl 0.5
#
#     cd ../
# done
# cd ../ # cd to ablation_study/

# #Ablate 80% #
# echo "TALOS: Ablate 80%"
#
# mkdir -p ablate_80/
# cd ablate_80
#
# for i in $(seq 1 10); do
#     mkdir -p "run_$i"/
#     cd "run_$i"/
#     rm -rf *
#
#     python $cwd/trajopt_parallel.py --exp chimney --robot talos --dz 3.0 --abl 0.8
#
#     cd ../
# done
# cd ../ # cd to ablation_study/
#
# cd ../ # cd to results/

