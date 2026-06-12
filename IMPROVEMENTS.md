# Improvements made

The project was improved without making the code structure more complicated.

## Beginner-friendly changes

- Rewrote the README so it explains the project more clearly.
- Added a simpler quickstart using generated sample data.
- Added clearer notes about what the project is and is not suitable for.
- Kept the same folder structure and the same main commands.

## Code quality changes

- Added validation for missing columns in `train.csv`.
- Added validation for `sample_frac`.
- Shuffled the dataset before the train/validation/test split.
- Calculated class weights from the training data instead of using a fixed value.
- Added `--freeze-base` for faster CPU demos.
- Rejected whitespace-only API inputs.
- Added a response model for batch predictions.
- Added validation for invalid prediction thresholds.

## Practical fixes

- Removed unused dependencies from `requirements.txt`.
- Added `curl` to the Docker image because the Docker healthcheck uses it.
- Added command-line arguments to the sample-data generator.
- Removed generated files, macOS metadata, and the embedded `.git` folder from the cleaned zip.
