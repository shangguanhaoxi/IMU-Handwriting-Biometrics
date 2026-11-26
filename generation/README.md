# Generation

This directory contains the generative-model research code. It is kept separate from the existing omc/ recognition implementation and verification/ experiments.

## LRF-IMU upstream implementation

The official public LRF-IMU repository is linked as a Git submodule:

- Upstream: https://github.com/aminsens/LRF-IMU
- Local path: generation/lrf_imu/
- Pinned upstream commit: ba333680db97a7bfeaf3e3540274b53dfb9d99c7
- Paper: Amin Rezaei, Morten Kjærgaard, and Jasper Schipperijn, *A latent rectified flow approach to generate synthetic wearable data – a LABDA solution*, Machine Learning: Health (2026), https://doi.org/10.1088/3049-477X/ae91ef

The upstream repository does not declare a repository license at the pinned commit. To avoid copying and redistributing source files without an explicit license, this project records the upstream source as a submodule rather than vendoring its files. Check the upstream licensing/permission before redistributing modified upstream code.

### Clone and initialize the source

For a fresh clone:

~~~bash
git clone --recurse-submodules https://github.com/shangguanhaoxi/IMU-Handwriting-Biometrics.git
cd IMU-Handwriting-Biometrics
~~~

For an existing checkout after pulling the parent repository:

~~~bash
git submodule sync --recursive
git submodule update --init --recursive
~~~

Install the upstream training dependencies in the intended Python environment:

~~~bash
cd generation/lrf_imu
python -m pip install -e ".[training]"
~~~

The upstream package supports Python 3.10+. The repository's own README and reproducibility documents are the source of truth for its environment and commands.

## Pretrained six-channel checkpoints

The checkpoint files are hosted separately from the GitHub source repository on Hugging Face:

- Model repository: https://huggingface.co/Aminrezaei/LRF-IMU
- Six-channel configuration: configs/six_channel_160_40.yaml
- Fold-specific matched pairs: checkpoints/six_channel/fold_XX/{vae.pt,flow.pt,manifest.json}
- The model card lists folds 01, 02, 03, 05, 08, 09, 10, 11, 12, 13, 14, 16.

Do not commit downloaded checkpoints or generated samples to this Git repository. Store them under an ignored experiment-output/checkpoint directory on AutoDL.

Example download of one matched six-channel fold pair (choose the fold deliberately; the fold IDs refer to REALDISP participants, not this handwriting dataset):

~~~bash
python -m pip install -U huggingface_hub
hf download Aminrezaei/LRF-IMU \
  model_index.json \
  configs/six_channel_160_40.yaml \
  checkpoints/six_channel/fold_01/manifest.json \
  checkpoints/six_channel/fold_01/vae.pt \
  checkpoints/six_channel/fold_01/flow.pt \
  --local-dir generation/checkpoints/lrf_imu/six_channel_fold_01
~~~

## Important transfer-learning constraints

The released six-channel model was trained on REALDISP right-thigh wearable IMU data: 50 Hz, 160-sample windows, six channels (three accelerometer + three gyroscope), a 48-by-40 latent representation, and four activity classes. Its activity labels are not handwriting-character labels.

For this project, treat the released weights only as candidate initialization weights. Before fine-tuning the Flow model, validate the VAE reconstruction on our six-axis handwriting signals. The upstream public VAE/checkpoint geometry is tied to 160 samples; changing the window length to 256 is not a drop-in operation. The activity-conditioning interface also needs adaptation for character-conditioned generation. Preserve the upstream checkpoints and configs unchanged; implement project-specific adapters and experiments outside the submodule unless upstream permission and a deliberate fork are established.

The upstream model card documents unresolved differences between the paper description and historical checkpoint configuration (exact_paper_reproduction=false). Read MODEL_CARD.md, REPRODUCIBILITY.md, and docs/KNOWN_DISCREPANCIES.md before interpreting a result as an exact paper reproduction.
