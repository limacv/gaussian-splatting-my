# python train.py \
#     -s /home/lmaag/xgpu-scratch/lmaag/data/zhu/ \
#     --model_path output/test \
#     -r 2 \
#     --test_iterations 10 100 1000 7000 30000 \
#     --checkpoint_iterations 7000 30000 

python train.py \
    -s /home/lmaag/xgpu-scratch/lmaag/data/zhu/ \
    --model_path output/test_normal_0 \
    --optimize_normal \
    -r 2 \
    --test_iterations 10 100 1000 7000 30000 \
    --checkpoint_iterations 7000 30000 