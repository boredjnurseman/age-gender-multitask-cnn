# Age and gender-label multi-task CNNs

This is a small computer-vision portfolio project about a practical modelling question: does ImageNet transfer learning help when a multi-task face model has only 5,000 task-specific training images?

Model A performed better on the recorded evaluation. It is a custom residual CNN that achieved an age MAE of 6.5747 years, binary gender-label accuracy of 0.901, and combined loss of 2.8543. Model B, a ResNet50V2 transfer-learning model, recorded 7.2968 years, 0.889, and 2.962 respectively. These are results for one split, checkpoint policy and training run; they are not a universal ranking of the two approaches.

The project keeps the comparison reproducible and makes the inference path reusable. It does not publish the UTKFace images used for training, and it does not treat the binary label as a person's gender identity.

## Models and recorded results

Both models take 128 × 128 RGB images scaled to `[0, 1]` and share a two-head design: a linear age regressor and a sigmoid output for the dataset-coded binary label.

| Model | Architecture and training | Age MAE (years) | Binary gender-label accuracy | Combined loss |
|---|---|---:|---:|---:|
| Model A | Custom residual CNN | 6.5747 | 0.901 | 2.8543 |
| Model B | ResNet50V2; frozen backbone, then fine-tuned final 30 layers | 7.2968 | 0.889 | 2.962 |

Model A uses Huber loss with `delta=6.5` for age and loss weights of `3.0` for gender and `0.30` for age. Model B uses the same task weights, Huber `delta=1.0`, and maps the image batch to `[-1, 1]` before ResNet50V2. The detailed scope and caveats are in [MODEL_CARD.md](MODEL_CARD.md).

## Public model files

The two saved models are hosted separately on public Google Drive links. The notebook and `age_gender_cnn.inference.download_models` cache them locally under the ignored `models/` directory.

| Model | File | Approximate size | Public link |
|---|---|---:|---|
| Model A | `age_gender_A.keras` | 29 MB | [download Model A](https://drive.google.com/uc?id=1AfnCchElx08FN0jGGLz91tl19YKWVP09) |
| Model B | `age_gender_B.keras` | 217 MB | [download Model B](https://drive.google.com/uc?id=1TdKyD8Bo7tByfuJxQbJUbnuak4tERydN) |

These files are supplied for this portfolio's research and demonstration workflow. Their availability does not change the terms of the training data or any image used for inference.

## Installation

Python 3.10–3.13 is supported. A local environment can be created with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install TensorFlow when you want to load the saved models or run the optional training path:

```bash
python -m pip install -e ".[dev,tensorflow]"
```

Run the ordinary offline test suite with:

```bash
MPLBACKEND=Agg PYTHONPATH=src python -m pytest -m "not integration" -q
```

## Notebook

Open `notebooks/Age_Gender_Multitask_CNN.ipynb` in Jupyter or Colab. The default configuration is reference mode:

```python
RUN_TRAINING_PIPELINE = False
DATA_DIR = None
MODEL_CACHE_DIR = None
TRAINING_OUTPUT_DIR = None
```

Reference mode reads the extracted metrics and curve figures in `artifacts/reference/`. It does not need `train_val`, does not call `fit`, and does not create checkpoints. The notebook can run through its narrative and plots without downloading the model files; the photo-upload cell downloads them when it is used.

To retrain, set `RUN_TRAINING_PIPELINE = True` and point `DATA_DIR` at an authorised local or mounted copy of `train_val`. The pipeline recreates the seeded 80/20 split, trains Model A, trains the frozen and fine-tuned phases of Model B, and writes new checkpoints below `TRAINING_OUTPUT_DIR`. Do not use this mode with data you are not permitted to process.

## Try the inference path

The notebook accepts exactly one permitted photograph. In Colab, use the upload prompt. In a local notebook, set `LOCAL_IMAGE_PATH` in the tagged interactive cell. The same image is decoded, passed through the face detector, cropped using the largest detected face or a visible centre-crop fallback, and sent to both models.

The result shows the original image, the crop actually supplied to the models, and the two predictions side by side. The sigmoid value is a dataset-coded binary gender-label score, not calibrated confidence and not an identity judgement. Uploaded files remain in the notebook session; this repository adds no remote storage or analytics. Duplicate upload bytes are rejected by the surrounding application workflow when that workflow is used.

The reusable package entry point is `age_gender_cnn.inference.predict_photo`. A Gradio or Streamlit application is a possible future extension, but it is not currently implemented.

## Data and licensing

The training subset is UTKFace, whose filenames encode labels. The images are not included in this repository. See [data/README.md](data/README.md) for the filename convention, source terms and local setup. The public augmentation figure uses the separately licensed [Simu Liu on The Beaverton](assets/README.md) image under CC BY 3.0.

Original code and documentation are released under the [MIT licence](LICENSE). That licence does not cover UTKFace, the trained model files or the CC BY 3.0 example photograph.

## Repository structure

```text
artifacts/reference/       Extracted metrics, provenance and learning curves
assets/                    Licensed example image and attribution
data/README.md             Private data setup and source terms
notebooks/                 Executed reference narrative and tagged demo cells
src/age_gender_cnn/        Data, models, training, plotting and inference modules
tests/                    Offline unit, notebook and hygiene tests
tools/                    Reference-evidence extraction utility
MODEL_CARD.md              Scope, evaluation and risk information
```

## Limitations

The evaluation uses one 5,000-image subset and one validation split. UTKFace labels include an estimated age and a reductive binary gender field; neither should be read as a complete description of a person. Subgroup performance was not measured, so the aggregate accuracy and MAE do not establish comparable performance across age, skin tone, pose, lighting or other groups. Face detection, crop choice and domain shift can also change the output.

The models are for education, method comparison and non-consequential experimentation with permitted images. They are not suitable for identity claims, surveillance, demographic measurement, employment, access decisions, policing, healthcare or any other consequential use.
