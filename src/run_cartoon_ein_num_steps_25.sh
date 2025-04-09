# Baseline
# python edit.py  --source_prompt "" \
#                 --target_prompt "a cartoon style Albert Einstein raising his left hand " \
#                 --guidance 2 \
#                 --source_img_dir 'examples/source/cartoon.jpg' \
#                 --num_steps 25 \
#                 --inject 2 \
#                 --name 'flux-dev' \
#                 --output_prefix 'rf_solver' \
#                 --offload \
#                 --output_dir 'examples/edit-result/cartoon/' 

# Fast Editing
python edit.py  --source_prompt "" \
                --target_prompt "a cartoon style Albert Einstein raising his left hand " \
                --guidance 2 \
                --source_img_dir 'examples/source/cartoon.jpg' \
                --num_steps 25 \
                --inject 1 \
                --name 'flux-dev' \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --sampling_strategy 'diffusion_free_noise' \
                --output_prefix 'diffusion_free_noise' \
                --offload \
                --output_dir 'examples/edit-result/cartoon/' 

# Better Instruction Following
python edit.py  --source_prompt "" \
                --target_prompt "a cartoon style Albert Einstein raising his left hand " \
                --guidance 2 \
                --source_img_dir 'examples/source/cartoon.jpg' \
                --num_steps 25 \
                --inject 1 \
                --name 'flux-dev' \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --sampling_strategy 'diffusion_free_noise' \
                --output_prefix 'diffusion_free_noise' \
                --offload \
                --output_dir 'examples/edit-result/cartoon/' 