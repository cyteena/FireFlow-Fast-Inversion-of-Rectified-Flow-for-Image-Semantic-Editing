python edit.py  --source_prompt "A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above." \
                --target_prompt "A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above." \
                --guidance 1 \
                --source_img_dir 'examples/source/boy.jpg' \
                --num_steps 8 \
                --offload \
                --inject 0 \
                --sampling_strategy 'diffusion_free_noise' \
                --output_prefix 'diffusion_free_noise' \
                --output_dir 'examples/edit-result/dog' 

python edit.py  --source_prompt "A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above." \
                --target_prompt "A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above." \
                --guidance 1 \
                --source_img_dir 'examples/source/boy.jpg' \
                --num_steps 8 \
                --offload \
                --inject 0 \
                --sampling_strategy 'reflow' \
                --output_prefix 'reflow' \
                --output_dir 'examples/edit-result/dog' 

python edit.py  --source_prompt "A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above." \
                --target_prompt "A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above." \
                --guidance 1 \
                --source_img_dir 'examples/source/boy.jpg' \
                --num_steps 8 \
                --offload \
                --inject 0 \
                --sampling_strategy 'rf_solver' \
                --output_prefix 'rf_solver' \
                --output_dir 'examples/edit-result/dog' 

python edit.py  --source_prompt "A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above." \
                --target_prompt "A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above." \
                --guidance 1 \
                --source_img_dir 'examples/source/boy.jpg' \
                --num_steps 8 \
                --offload \
                --inject 0 \
                --sampling_strategy 'rf_midpoint' \
                --output_prefix 'rf_midpoint' \
                --output_dir 'examples/edit-result/dog' 
