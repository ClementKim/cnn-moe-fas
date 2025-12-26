#!/bin/bash

# conda activate cnn-moe
source .venv/bin/activate

for cls_weight in 0.5 0.6 0.7 0.8 0.9 1.0; do
    for arc_weight in 0.5 0.6 0.7 0.8 0.9 1.0; do
        for expert in 4 8; do
            for seed in 42 43 44 777 911; do
                for num in {1..3}; do
                    python3 main_moe.py --seed ${seed} --num_experts ${expert} --cls_weight ${cls_weight} --arc_weight ${arc_weight} > logs/cnn_moe/seed_${seed}_expert_${expert}_cls_${cls_weight}_arc_${arc_weight}_${num}_log.txt
                    python3 main_cnn.py --seed ${seed} --cls_weight ${cls_weight} --arc_weight ${arc_weight} > logs/cnn/seed_${seed}_cls_${cls_weight}_arc_${arc_weight}_${num}_log.txt
                done
            done
        done
    done
done