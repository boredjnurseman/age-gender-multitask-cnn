# Model card: age and gender-label multi-task CNNs

## Summary

This repository compares two Keras multi-task models for age regression and prediction of the binary gender label encoded by the UTKFace dataset: a custom residual CNN and a ResNet50V2 transfer model. The models were trained and evaluated as a portfolio experiment, not as a validated demographic or identity system.

The recorded selected validation result favoured Model A under the documented split and checkpoint policy. The numbers below come from the structured outputs extracted from the final executed source notebook and are not a claim about performance on a new population.

## Intended use

The models are intended for:

- reviewing a reproducible comparison between a purpose-built residual CNN and transfer learning;
- teaching or discussing multi-task loss weighting, checkpoint selection and image preprocessing;
- non-consequential experimentation on photographs that the user is permitted to process.

## Out-of-scope use

Do not use these models to infer gender identity, identify people, make decisions about employment, education, housing, insurance, healthcare, policing or access to services, or to support surveillance or demographic monitoring. The output is not calibrated confidence, proof of identity or a measurement of an intrinsic attribute.

## Data and label semantics

The source experiment used a private 5,000-image subset of [UTKFace](https://susanqq.github.io/UTKFace/). UTKFace filenames encode age, a binary gender field, race and collection time. The gender field follows the source convention of `0` for male and `1` for female. In this project it is described as a dataset-coded binary gender label because that field is reductive and does not represent a person's gender identity.

The dataset's age annotations are estimated from appearance and checked by a human annotator. The model therefore predicts a source label from an image; it does not observe chronological age directly. The source page states that UTKFace is available for non-commercial research only, that copyright belongs to the original owners, and that the dataset maintainers are not responsible for the images' content or meaning.

## Inputs and outputs

- Input: one colour image, face-cropped or centre-cropped to 128 × 128 pixels and scaled to `[0, 1]`.
- Age output: a scalar point estimate in years.
- Gender output: a sigmoid score mapped at 0.5 to the source labels `male` or `female`.
- The gender score is a model score for the dataset convention, not calibrated probability, gender identity or identity evidence.

## Architectures and training policy

| Model | Architecture | Training policy | Parameters |
|---|---|---|---:|
| Model A | Custom residual CNN with a convolutional stem, three residual blocks, a shared dense representation and two task heads | Trained with binary cross-entropy for the label, `Huber(delta=6.5)` for age, loss weights `gender_output=3.0` and `age_output=0.30` | 2,440,834 |
| Model B | ResNet50V2 backbone with global average pooling, shared dense representation and two task heads | ImageNet backbone frozen first, then the last 30 layers of the backbone fine-tuned at a lower learning rate; Batch Normalisation layers remain frozen; `Huber(delta=1.0)` and the same task weights | 24,123,394 |

Both models use the same 128 × 128 RGB input contract. Model B also rescales the batch from `[0, 1]` to `[-1, 1]` before the ResNet50V2 backbone. The default notebook setting is `RUN_TRAINING_PIPELINE = False`; training is never enabled merely because data or a previous kernel variable exists.

## Evaluation

The selected checkpoint metrics are:

| Model | Selected checkpoint | Age MAE (years) | Gender-label accuracy | Combined validation loss |
|---|---|---:|---:|---:|
| Model A | `best_val_loss` | 6.5747 | 0.901 | 2.8543 |
| Model B | `tuned_best_val_loss` | 7.2968 | 0.889 | 2.962 |

The Model A result is better on all three recorded values for this validation comparison. Model B improved after fine-tuning, but did not overtake Model A. An older submitted report recorded Model A as 6.8488 age MAE and 0.906 gender accuracy; those values are retained as a provenance discrepancy and are not mixed into the table above.

## Model files

The public saved models are hosted on Google Drive:

- Model A, `age_gender_A.keras`, approximately 29 MB: [1AfnCchElx08FN0jGGLz91tl19YKWVP09](https://drive.google.com/uc?id=1AfnCchElx08FN0jGGLz91tl19YKWVP09)
- Model B, `age_gender_B.keras`, approximately 217 MB: [1TdKyD8Bo7tByfuJxQbJUbnuak4tERydN](https://drive.google.com/uc?id=1TdKyD8Bo7tByfuJxQbJUbnuak4tERydN)

The approximate sizes identify the expected public archives; a downloaded file is checked for a valid non-trivial `.keras` ZIP archive before loading. The files are not committed to the repository.

## Limitations and fairness risks

- The experiment uses one 5,000-image subset, one seeded 80/20 split and one checkpoint-selection policy. The comparison may change under another split, sample or training budget.
- UTKFace contains images collected from the internet, estimated labels and a binary gender field that does not cover gender diversity.
- No subgroup evaluation was performed. Aggregate MAE and accuracy cannot establish equal performance across age groups, skin tones, gender expression, pose, disability, lighting or image quality.
- The models predict from appearance and may reproduce sampling, annotation and face-detection bias. A correct source label in the dataset would still not make a real-world decision appropriate.
- Haar-cascade detection, the largest-face rule and centre-crop fallback can change which pixels reach the models.
- The scores are not calibrated and no uncertainty estimate is provided. One decimal place for age is display formatting, not meaningful precision.

These limitations rule out consequential use. A responsible follow-up would report subgroup and age-stratified errors on authorised data, test repeated fixed splits, examine calibration and document consent and retention controls before any broader experiment.

## Privacy

Only process photographs for which you have permission. The notebook keeps uploaded files in the local or hosted notebook session and the package does not add remote storage, logging or analytics. Delete local copies and notebook outputs when they are no longer needed. Do not commit photographs, raw UTKFace data or downloaded model files.

## Licensing and provenance

The original code and documentation are MIT-licensed. The MIT licence excludes UTKFace, the separately hosted trained model files and the CC BY 3.0 example photograph. The example image and its modifications are documented in [assets/README.md](assets/README.md).
