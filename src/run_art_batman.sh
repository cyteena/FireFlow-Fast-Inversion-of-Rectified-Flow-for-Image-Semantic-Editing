# Baseline
# CUDA_VISIBLE_DEVICES=7 python edit.py  --source_prompt "" \
#                 --target_prompt "a vivid depiction of the Batman, featuring rich, dynamic colors,  and a blend of realistic and abstract elements with dynamic splatter art." \
#                 --guidance 2 \
#                 --source_img_dir 'examples/source/art.jpg' \
#                 --num_steps 25  \
#                 --inject 5 \
#                 --offload \
#                 --name 'flux-dev'  \
#                 --output_prefix 'rf_solver' \
#                 --output_dir 'examples/edit-result/art/' 

# Fast Editing
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "" \
                --target_prompt "a vivid depiction of the Batman, featuring rich, dynamic colors,  and a blend of realistic and abstract elements with dynamic splatter art." \
                --guidance 2 \
                --source_img_dir 'examples/source/art.jpg' \
                --num_steps 8  \
                --inject 1 \
                --offload \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --name 'flux-dev'  \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow' \
                --output_dir 'examples/edit-result/art/'

# Better Instruction Following
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "" \
                --target_prompt "a vivid depiction of the Batman, featuring rich, dynamic colors,  and a blend of realistic and abstract elements with dynamic splatter art." \
                --guidance 2 \
                --source_img_dir 'examples/source/art.jpg' \
                --num_steps 8  \
                --inject 1 \
                --offload \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --name 'flux-dev'  \
                --reuse_v 0 \
                --editing_strategy 'add_q' \
                --sampling_strategy 'diffusion_free_noise' \
                --output_prefix 'fireflow_add_q' \
                --output_dir 'examples/edit-result/art/' 

# denoise_strategies = {
#     'reflow' : denoise,
#     'rf_solver' : denoise_rf_solver,
#     'fireflow' : denoise_fireflow,
#     'rf_midpoint' : denoise_midpoint,
#     'ada_solver' : denoise_adaptive,
#     'diffusion_free_noise' : denoist_Ralston
# }
