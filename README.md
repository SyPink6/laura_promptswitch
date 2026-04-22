<a name="depend"/>

## Dependencies
Our model was developed and evaluated using the following package dependencies:
- PyTorch 1.8.2
- OpenCV 4.7.0
- transformers 4.26.1
- timm
- tensorboard
- pandas

## Setup
Install the Python dependencies from `requirements.txt`.

The repository only includes annotations and split files under `data/`. You still need to download the actual dataset videos and point `--videos_dir` to the correct folder.

The code loads the HuggingFace checkpoint `openai/clip-vit-base-patch32` for both the tokenizer and model. Make sure the machine can access HuggingFace on the first run, or that the checkpoint is already cached locally.

## Minimal Commands
Train with the default `PromptCLIP` model:

```bash
python train.py --exp_name run1 --dataset_name MSRVTT --videos_dir data/MSRVTT/vids
```

Evaluate the best checkpoint:

```bash
python test.py --exp_name run1 --dataset_name MSRVTT --videos_dir data/MSRVTT/vids --load_epoch -1
```

If you want to use the transformer pooling baseline explicitly:

```bash
python train.py --exp_name run1_tf --arch clip_transformer --dataset_name MSRVTT --videos_dir data/MSRVTT/vids
```

## Notes
- `--num_frames` and `--num_test_frames` must both be divisible by `--num_prompts`.
- If `--videos_dir` does not exist, the code now fails early with a clear error message instead of crashing inside OpenCV.
- `test.py` is the evaluation entry point, not a unit test file.
