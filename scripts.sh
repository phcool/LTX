CUDA_VISIBLE_DEVICES=0 python -m ltx_pipelines.ti2vid_two_stages \
  --checkpoint-path /nfs/hanpeng/huggingface/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --distilled-lora /nfs/hanpeng/huggingface/models/LTX-2.3/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
  --spatial-upsampler-path /nfs/hanpeng/huggingface/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --gemma-root /nfs/hanpeng/huggingface/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "a close-up of a cat walking on a concrete surface. The cat has a mix of gray, black, and orange fur with distinct stripes. Its eyes are large and round, with a greenish-yellow color. The cat's nose is pink, and it has white whiskers. The background is slightly blurred, but it appears to be an outdoor setting with some gravel and patches of grass. The cat is moving forward, and its tail is raised slightly. The lighting is natural, suggesting it might be daytime." \
  --output-path output/cat_sft_6.mp4 \
  --lora /nfs/hanpeng/LTX-2/pexels/outputs/checkpoints/lora_weights_step_00300.safetensors 5

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m ltx_pipelines.ti2vid_two_stages \
  --checkpoint-path /nfs/hanpeng/huggingface/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --distilled-lora /nfs/hanpeng/LTX-2/pexels/outputs/checkpoints 0.8 \
  --spatial-upsampler-path /nfs/hanpeng/huggingface/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --gemma-root /nfs/hanpeng/huggingface/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "A cinematic close-up of a cat walking through neon-lit rain at night, detailed fur, shallow depth of field, smooth camera movement. The sound of the rain is a subtle background noise." \
  --output-path output/new_cat.mp4

python -m ltx_pipelines.distilled \
  --distilled-checkpoint-path /nfs/hanpeng/huggingface/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors \
  --spatial-upsampler-path /nfs/hanpeng/huggingface/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --gemma-root /nfs/hanpeng/huggingface/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "a close-up of a cat walking on a concrete surface. The cat has a mix of gray, black, and orange fur with distinct stripes. Its eyes are large and round, with a greenish-yellow color. The cat's nose is pink, and it has white whiskers. The background is slightly blurred, but it appears to be an outdoor setting with some gravel and patches of grass. The cat is moving forward, and its tail is raised slightly. The lighting is natural, suggesting it might be daytime." \
  --output-path output/cat-distilled.mp4

CUDA_VISIBLE_DEVICES=2 python -m ltx_pipelines.distilled \
  --distilled-checkpoint-path /nfs/hanpeng/huggingface/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --spatial-upsampler-path /nfs/hanpeng/huggingface/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --gemma-root /nfs/hanpeng/huggingface/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --lora /nfs/hanpeng/LTX-2/packages/ltx-trainer/outputs/ltx2_22b_offline_teacher_train_h100x8/checkpoints/lora_weights_step_02000.safetensors 2 \
  --prompt "a close-up of a cat walking on a concrete surface. The cat has a mix of gray, black, and orange fur with distinct stripes. Its eyes are large and round, with a greenish-yellow color. The cat's nose is pink, and it has white whiskers. The background is slightly blurred, but it appears to be an outdoor setting with some gravel and patches of grass. The cat is moving forward, and its tail is raised slightly. The lighting is natural, suggesting it might be daytime." \
  --output-path output/my-lora-on-distilled_2.mp4

python -m ltx_pipelines.ic_lora \
  --distilled-checkpoint-path /nfs/hanpeng/huggingface/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors \
  --spatial-upsampler-path /nfs/hanpeng/huggingface/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --gemma-root /nfs/hanpeng/huggingface/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --lora /nfs/hanpeng/huggingface/models/ic-lora/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors 1.0 \
  --video-conditioning /nfs/hanpeng/LTX-2/assets/videos/man.mp4 1.0 \
  --prompt "A man walking in the street, natural lighting, cinematic." \
  --output-path output/man.mp4 \
  --images /nfs/hanpeng/LTX-2/assets/images/background_832x480.png 0 1.0

uv run python packages/ltx-trainer/scripts/caption_videos_multi_gpu.py \
  pexels_9k/videos \
  --output pexels_9k/dataset_captioned.json \
  --gpus 0,1,2,3,4,5,6,7 \
  --no-audio

uv run python packages/ltx-trainer/scripts/process_dataset.py \
  pexels/dataset_captioned.json \
  --resolution-buckets "768x448x49" \
  --model-path /nfs/hanpeng/huggingface/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --text-encoder-path /nfs/hanpeng/huggingface/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --output-dir pexels/.precomputed \

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch \
  --config_file packages/ltx-trainer/configs/accelerate/ddp.yaml \
  packages/ltx-trainer/scripts/train.py \
  pexels/train_config.yaml