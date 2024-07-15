{

#######################################################################
##  Optimize Normal
#######################################################################
python train.py \
    -s /root/public/gss/VPS05_kevin_relit02_0_1_frm38/ \
    --model_path /root/public/gss/results/kevin_frm38_relit_withmask \
    --test_iterations 10 100 1000 7000 30000 \
    --checkpoint_iterations 7000 30000 

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

# for dataset in apartment; do

# 3w
# python train.py \
#     -s /cpfs01/shared/pjlab-lingjun-landmarks/eyeful/$dataset \
#     --model_path /cpfs01/user/mali1/output/eyeful_${dataset}_3view_nodistort_jpeg2k \
#     --eyeful_loadcamera "19,20,21,17" \
#     --eyeful_subdir images-jpeg-2k \
#     --test_iterations 10 100 1000 7000 10000 25000 30000 \
#     --checkpoint_iterations 7000 30000 \
#     --eval

# 6w
# python train.py \
#     -s /cpfs01/shared/pjlab-lingjun-landmarks/eyeful/$dataset \
#     --model_path /cpfs01/user/mali1/output/eyeful_${dataset}_jpeg2k_6w \
#     --resolution 2 \
#     --eyeful_subdir images-jpeg-2k \
#     --test_iterations 10 100 1000 7000 10000 20000 30000 40000 50000 60000 \
#     --checkpoint_iterations 15000 60000\
#     --save_iterations 15000 3000 60000\
#     --iterations 60000 \
#     --position_lr_max_steps 60000 \
#     --global_lr_scalar 0.5 \
#     --densify_until_iter 30000 \
#     --eval

# done

# lujiazui_9_huanqiu_tiejin
# lujiazui_9_jinmao_tiejin
# for jsonname in lujiazui_9_jinmao_tiejin; do
# python train.py \
#     -s /cpfs01/shared/pjlab-lingjun-landmarks/data/sanjiantao \
#     --model_path /cpfs01/user/mali1/output/hold6_${jsonname} \
#     --city_json transforms_${jsonname}_downsample5.json \
#     --city_loadcamerahold 6 \
#     --resolution 1 \
#     --test_iterations 10 100 1000 7000 10000 25000 30000 \
#     --checkpoint_iterations 7000 30000\
#     --eval
# done

exit 0
}
