# Baseline
# python edit.py  --source_prompt "" \
#                 --target_prompt "a vivid depiction of the Marilyn Monroe, featuring rich, dynamic colors,  and a blend of realistic and abstract elements with dynamic splatter art." \
#                 --guidance 2 \
#                 --source_img_dir 'examples/source/art.jpg' \
#                 --num_steps 15  \
#                 --inject 3 \
#                 --name 'flux-dev'  \
#                 --offload \
#                 --output_prefix 'rf_solver' \
#                 --output_dir 'examples/edit-result/art/'

# # Fast Editing
python edit.py  --source_prompt "" \
                --target_prompt "a vivid depiction of the Marilyn Monroe, featuring rich, dynamic colors,  and a blend of realistic and abstract elements with dynamic splatter art." \
                --guidance 2 \
                --source_img_dir 'examples/source/art.jpg' \
                --num_steps 15  \
                --inject 1 \
                --name 'flux-dev'  \
                --offload \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --sampling_strategy 'diffusion_free_noise' \
                --output_prefix 'diffusion_free_noise' \
                --output_dir 'examples/edit-result/art/' 

# Better Instruction Following
python edit.py  --source_prompt "" \
                --target_prompt "a vivid depiction of the Marilyn Monroe, featuring rich, dynamic colors,  and a blend of realistic and abstract elements with dynamic splatter art." \
                --guidance 2 \
                --source_img_dir 'examples/source/art.jpg' \
                --num_steps 15  \
                --inject 1 \
                --name 'flux-dev'  \
                --offload \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --reuse_v 0 \
                --editing_strategy 'add_q' \
                --sampling_strategy 'diffusion_free_noise' \
                --output_prefix 'diffusion_free_noise_add_q' \
                --output_dir 'examples/edit-result/art/' 
