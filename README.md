# Fourier-World

Fourier-World predicts future DINOv3 patch features from four observed video frames. The predictor takes a temporal real FFT at each patch, reads the resulting frequency components with a learned query, and adds the query update to the latest observation. The timed model conditions attention keys on the observed and requested time intervals. Training and validation operate on Kinetics-700 videos.

## Installation

Use Linux and Python 3.11. The training environment uses PyTorch 2.6.0, torchvision 0.21.0, and CUDA 12.6. A CUDA GPU with bfloat16 support is recommended for training. The scripts also run on CPU.

```bash
git clone https://github.com/Tiabet/Fourier-World.git
cd Fourier-World
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

For a CPU environment, use `https://download.pytorch.org/whl/cpu` as the PyTorch index in the first install command.

## Data and backbone

Download the 700 Kinetics-700 (2020) training archives into a single directory. The loader streams MP4 members directly from `k700_train_001.tar.gz` through `k700_train_700.tar.gz`. The exclusion list contains one archive member path per line. [Kinetics download instructions](https://github.com/cvdfoundation/kinetics-dataset) · [Training shard URLs](https://s3.amazonaws.com/kinetics/700_2020/train/k700_2020_train_path.txt)

```bash
mkdir -p data/k700
wget -c -P data/k700 -i https://s3.amazonaws.com/kinetics/700_2020/train/k700_2020_train_path.txt
wget -O data/kinetics_excluded.txt https://raw.githubusercontent.com/amazon-far/deltatok/e1ec9b5d9aa0fdb7844819cedab535b4b8109260/datasets/kinetics_excluded.txt
```

Request access to DINOv3 ViT-B/16 and authenticate with `hf auth login`. The `--backbone` argument accepts the model ID below or the path to a downloaded model snapshot. [DINOv3 weights and access](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)

## Training

```bash
torchrun --standalone --nproc-per-node=4 train.py \
  --tars data/k700 \
  --exclude-list data/kinetics_excluded.txt \
  --backbone facebook/dinov3-vitb16-pretrain-lvd1689m \
  --output runs/k700
```

Each video supplies one five-frame window from an irregular eight-frame draw. All frames in a clip share the same crop and horizontal flip. The first four feature grids form the context, and the fifth is the H1 target. Only the predictor receives gradients, using full-feature Smooth L1 loss with beta 0.1.

The default run uses 512-pixel square crops, 60,000 optimizer updates, and a global batch of 1,024 clips. Gradient accumulation preserves the global batch when you reduce `--microbatch` to fit GPU memory. Use `python train.py` with the same arguments for a single device.

The trainer saves `progress.pt` at optimizer boundaries and renames the final checkpoint to `last.pt`. Repeating the command for an interrupted output directory restores model weights, optimizer state, and the step counter. The streaming sampler starts a new deterministic sequence from the resumed step.

## Validation

The first eight eligible members of each shard are reserved for validation. By default, the evaluator selects 256 evenly spaced shards and decodes one held-out clip per shard. It predicts H1 directly and H3 by feeding the first two predictions back into the model. It computes feature-space Smooth L1 for the predictor and for repeating the last observed feature.

```bash
python evaluate.py \
  --checkpoint runs/k700/last.pt \
  --tars data/k700 \
  --exclude-list data/kinetics_excluded.txt \
  --backbone facebook/dinov3-vitb16-pretrain-lvd1689m
```

Pass `--samples 1` for a one-clip check. The evaluator reads the frame size, sampling seed, and timing configuration from the checkpoint.

## Predict from a clip

Given four RGB images of the same size, this example encodes their spatial patches and forecasts at three specified times in seconds. Both image dimensions must be multiples of 16. The returned tensor has shape `[batch, future steps, patches, 768]`.

```python
import numpy as np
import torch
from PIL import Image

from fourier_world import TimedFourierWorld
from fourier_world.backbone import DINOv3DenseFeatures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("runs/k700/last.pt", map_location="cpu", weights_only=True, mmap=True)
model = TimedFourierWorld(reference_dt=checkpoint["validation_config"]["reference_dt"])
model.load_state_dict(checkpoint["model"])
model.eval().to(device)
del checkpoint

encoder = DINOv3DenseFeatures("facebook/dinov3-vitb16-pretrain-lvd1689m").eval().to(device)
images = []
for name in ["frame0.png", "frame1.png", "frame2.png", "frame3.png"]:
    with Image.open(name) as image:
        images.append(np.asarray(image.convert("RGB")))
frames = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
observed = torch.tensor([[0.0, 0.12, 0.40, 0.60]], device=device)
grid = (frames.shape[-2] // encoder.patch_size, frames.shape[-1] // encoder.patch_size)

with torch.inference_mode():
    state = model.initialize(encoder(frames), observed)
    predictions = []
    for time in [0.8, 1.0, 1.2]:
        prediction, state = model.step(state, future_timestamp=time, spatial_shape=grid)
        predictions.append(prediction)
    forecast = torch.stack(predictions, dim=1)
```
