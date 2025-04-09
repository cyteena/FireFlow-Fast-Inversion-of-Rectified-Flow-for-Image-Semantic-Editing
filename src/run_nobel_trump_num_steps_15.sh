# Baseline
# python edit.py  --source_prompt "" \
#                 --target_prompt "A minimalistic line-drawing portrait of Donald Trump with black lines and brown shadow" \
#                 --guidance 2.5 \
#                 --source_img_dir 'examples/source/nobel.jpg' \
#                 --num_steps 15  \
#                 --inject 3 \
#                 --name 'flux-dev' \
#                 --offload \
#                 --output_prefix 'rf_solver' \
#                 --output_dir 'examples/edit-result/nobel/' 

# Fast Editing
python edit.py  --source_prompt "" \
                --target_prompt "A minimalistic line-drawing portrait of Donald Trump with black lines and brown shadow" \
                --guidance 3 \
                --source_img_dir 'examples/source/nobel.jpg' \
                --num_steps 15  \
                --inject 1 \
                --name 'flux-dev' \
                --offload \
                --start_layer_index 0 \
                --end_layer_index 37 \
                --sampling_strategy 'diffusion_free_noise' \
                --output_prefix 'diffusion_free_noise' \
                --output_dir 'examples/edit-result/nobel/' 
