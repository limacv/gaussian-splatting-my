{

#######################################################################
##  Optimize Normal
#######################################################################
# python train.py \
#     -s /home/lmaag/xgpu-scratch/lmaag/data/zhu/ \
#     --model_path output/test \
#     -r 2 \
#     --test_iterations 10 100 1000 7000 30000 \
#     --checkpoint_iterations 7000 30000 

# python train.py \
#     -s /home/lmaag/xgpu-scratch/lmaag/data/zhu/ \
#     --model_path output/test_normal_mod13 \
#     --optimize_normal \
#     -r 2 \
#     --test_iterations 10 100 1000 7000 30000 \
#     --checkpoint_iterations 7000 30000 

# python train.py \
#     -s /home/lmaag/xgpu-scratch/lmaag/data/tandt/truck \
#     --model_path output/truck_mod12 \
#     --optimize_normal \
#     --test_iterations 10 100 1000 7000 30000 \
#     --checkpoint_iterations 7000 30000 

# python train.py \
#     -s /cpfs01/user/mali1/data/tandt/truck \
#     --model_path /cpfs01/user/mali1/output/truck_modifier12 \
#     --optimize_normal \
#     --test_iterations 10 100 1000 7000 30000 \
#     --checkpoint_iterations 7000 30000 

# python train.py \
#     -s /cpfs01/user/mali1/data/tandt/truck \
#     --model_path /cpfs01/user/mali1/output/truck_normfilter \
#     --optimize_normal \
#     --test_iterations 10 100 1000 7000 30000 \
#     --checkpoint_iterations 7000 30000 


#######################################################################
## Approximate Sorting
#######################################################################

# python train.py \
#     -s /cpfs01/user/mali1/data/tandt/truck \
#     --model_path /cpfs01/user/mali1/output/truck_base_eval \
#     --test_iterations 10 100 1000 7000 10000 20000 25000 30000 \
#     --checkpoint_iterations 7000 30000 \
#     --eval \
#     --backend original


# python train.py \
#     -s /cpfs01/user/mali1/data/tandt/truck \
#     --model_path /cpfs01/user/mali1/output/truck_my_eval \
#     --test_iterations 10 100 1000 7000 10000 20000 25000 30000 \
#     --checkpoint_iterations 7000 30000 \
#     --eval \
#     --backend my

python train.py \
    -s /cpfs01/shared/pjlab-lingjun-landmarks/eyeful/apartment \
    --model_path /cpfs01/user/mali1/output/eyeful_apartment_jpeg2k \
    --test_iterations 10 100 1000 7000 10000 20000 25000 30000 \
    --checkpoint_iterations 7000 30000 \
    --eval

exit 0
}