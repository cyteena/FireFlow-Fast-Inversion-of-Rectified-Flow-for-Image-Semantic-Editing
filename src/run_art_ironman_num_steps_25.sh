# # Fast Editing
python edit.py  --source_prompt "" \
                --target_prompt "a vivid depiction of the ironman, featuring rich, dynamic colors,  and a blend of realistic and abstract elements with dynamic splatter art." \
                --guidance 2 \
                --source_img_dir 'examples/source/art.jpg' \
                --num_steps 25  \
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
                --target_prompt "a vivid depiction of the ironman, featuring rich, dynamic colors,  and a blend of realistic and abstract elements with dynamic splatter art." \
                --guidance 2 \
                --source_img_dir 'examples/source/art.jpg' \
                --num_steps 25  \
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


# Better Instruction Following
python edit.py  --source_prompt "" \
                --target_prompt "a vivid depiction of the ironman, featuring rich, dynamic colors,  and a blend of realistic and abstract elements with dynamic splatter art." \
                --guidance 4 \
                --source_img_dir 'examples/source/art.jpg' \
                --num_steps 50  \
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


# python edit.py  --source_prompt "" \
#                 --target_prompt "a vivid depiction of the ironman, featuring rich, dynamic colors,  and a blend of realistic and abstract elements with dynamic splatter art." \
#                 --guidance 2 \
#                 --source_img_dir 'examples/source/art.jpg' \
#                 --num_steps 25  \
#                 --inject 5 \
#                 --name 'flux-dev'  \
#                 --offload \
#                 --start_layer_index 0 \
#                 --end_layer_index 37 \
#                 --reuse_v 0 \
#                 --editing_strategy 'add_q' \
#                 --sampling_strategy 'diffusion_free_noise' \
#                 --output_prefix 'diffusion_free_noise_add_q' \
#                 --output_dir 'examples/edit-result/art/'
