# Data setup

The training experiment used a private, authorised subset of UTKFace. The images are intentionally not included in this repository and must not be redistributed with it. Reference-mode notebook execution does not need this directory; `train_val` is required only when explicit retraining is enabled.

## UTKFace source and filename convention

Read the [official UTKFace page](https://susanqq.github.io/UTKFace/) before obtaining or using the data. The source describes the dataset as available for non-commercial research purposes only, notes that copyright belongs to the original image owners, and explains that its labels are estimates checked by a human annotator. Follow the official source terms rather than treating this repository as a new dataset licence.

Each image filename follows this pattern:

```text
[age]_[gender]_[race]_[date&time].jpg
```

The source defines `[age]` as an integer from 0 to 116, `[gender]` as `0` for male or `1` for female, `[race]` as the source's integer-coded category, and `[date&time]` as the collection timestamp. This project uses only age and the binary gender field, and calls the latter a dataset-coded label rather than a person's gender identity.

## Local training setup

Place the images in a local directory such as `train_val/` that you are authorised to use. Do not commit that directory. Set the notebook configuration explicitly:

```python
RUN_TRAINING_PIPELINE = True
DATA_DIR = "/path/to/your/authorised/train_val"
TRAINING_OUTPUT_DIR = "/path/to/a/local/training-output"
```

The data loader checks the directory and filename labels before allocating arrays. It then creates the deterministic 80/20 split used by the recorded experiment. Training writes checkpoints to the selected output directory; it does not overwrite the public saved models by default.

## Public augmentation example

The repository's displayed augmentation example is separate from UTKFace. It is “Simu Liu on The Beaverton” by The Beaverton, used under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). Source and modification details are in [assets/README.md](../assets/README.md). That photograph is not a substitute for the private training data, and its licence does not grant permission to redistribute UTKFace images.
