# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project uses a Random Forest Classifier to predict whether an individual's income exceeds $50K/year based on census data. The model uses one-hot encoding for categorical features and standard preprocessing for continuous features.

## Intended Use
The model is intended for educational purposes and to demonstrate building a scalable ML pipeline using Python, scikit-learn, and FastAPI. It predicts income categories from census features. It is not intended for real-world financial decisions.

## Training Data
The training data is derived from the Census Income Dataset, consisting of features such as age, education, occupation, workclass, marital status, race, sex, capital gain/loss, and hours-per-week.

## Evaluation Data
The model is evaluated on a hold-out test set (20% of the dataset) and performance is also calculated on slices of categorical features to identify potential disparities.

## Metrics
Overall performance on the test set:

- Precision: 0.7419
- Recall: 0.6384
- F1 Score: 0.6863

Additionally, performance was measured on slices of categorical features (such as: workclass, education) and saved to slice_output.txt.

## Ethical Considerations
Income prediction models can perpetuate biases in the dataset, particularly with features such as race, sex, and education. This model should not be used for real-world employment, credit, or financial decisions.

## Caveats and Recommendations
- The model performance varies across slices of categorical features.
- Consider additional preprocessing or fairness adjustments for sensitive features.
- The model is trained on historical U.S. census data and may not generalize to other populations or time periods.