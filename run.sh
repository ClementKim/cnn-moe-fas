#!/bin/bash

# conda activate js_dom
source .venv/bin/activate

for expert in 2 4 8 16 32 64 128 256 512 1024; do
    for seed in 42 43 44 777 911; do
        for num in {1..3}; do
            python3 main.py --seed ${seed} --num_experts ${expert} > logs/seed_${seed}_expert_${expert}_${num}_log.txt

        done
    done
done

deactivate
