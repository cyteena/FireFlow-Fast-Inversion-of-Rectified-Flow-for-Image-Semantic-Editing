# Baseline
# python edit.py  --source_prompt "A young boy is riding a brown horse in a countryside field, with a large tree in the background." \
#                 --target_prompt "A young boy is riding a camel in a countryside field, with a large tree in the background." \
#                 --guidance 2 \
#                 --source_img_dir 'examples/source/horse.jpg' \
#                 --num_steps 25  \
#                 --inject 3 \
#                 --offload \
#                 --name 'flux-dev' \
#                 --output_prefix 'rf_solver' \
#                 --output_dir 'examples/edit-result/horse/' 

# Fast Editing
python edit.py  --source_prompt "A young boy is riding a brown horse in a countryside field, with a large tree in the background." \
                --target_prompt "A young boy is riding a camel in a countryside field, with a large tree in the background." \
                --guidance 2 \
                --source_img_dir 'examples/source/horse.jpg' \
                --num_steps 25  \
                --inject 1 \
                --offload \
                --name 'flux-dev' \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --sampling_strategy 'diffusion_free_noise' \
                --output_prefix 'diffusion_free_noise' \
                --output_dir 'examples/edit-result/horse/'

# Add
python edit.py  --source_prompt "A young boy is riding a brown horse in a countryside field, with a large tree in the background." \
                --target_prompt "Two young boys are riding a brown horse in a countryside field, with a large tree in the background." \
                --guidance 2 \
                --source_img_dir 'examples/source/horse.jpg' \
                --num_steps 25  \
                --inject 1 \
                --offload \
                --name 'flux-dev' \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --sampling_strategy 'diffusion_free_noise' \
                --output_prefix 'diffusion_free_noise' \
                --output_dir 'examples/edit-result/horse/'
