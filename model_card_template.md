# Model Card

## Model Details
This model is a simple logistic regression classifier developed to illustrate the project and is not recommended for production use. The model predicts income levels based on various features derived from census data, classifying individuals into two income categories: whether they earn more or less than $50,000 annually. 
[Link to Model Page](https://archive.ics.uci.edu/dataset/2/adult)

## Intended Use
The model is intended for educational and research purposes in socio-economic studies and economic research. It should be applied only for demographic and income prediction tasks that align with the training data's representation.

## Training Data
The model was trained on the "Adult" dataset, also known as the "Census Income" dataset, from the 1994 US Census database. The dataset includes demographic and employment information, with attributes such as age, work class, education, occupation, and hours per week.

## Evaluation Data
Evaluation was performed on a held-out test set from the same Census Income dataset. This test set was not part of the training process and represents the model's performance on unseen data.

## Metrics
The model's performance is evaluated based on the following metrics:
- Precision: 0.714 (71.4%)
- Recall: 0.251 (25.1%)
- F1-Score: 0.372 (37.2%)

These metrics reflect the model's conservative predictions, with a higher emphasis on precision over recall.

The analysis of model performance across different categories reveals significant variations in precision, recall, and F1-scores, suggesting differential predictive accuracy based on socio-demographic factors such as workclass, education, marital status, race, sex, and native country. For example, the model performs better in predicting higher income levels for individuals in certain work classes like 'Self-emp-inc' and 'Local-gov', and those with higher educational qualifications such as 'Doctorate' and 'Masters'. This indicates a possible correlation in the training data between higher education and income levels. However, the model's predictive accuracy decreases for lower education levels and in certain work classes like 'Never-worked' or 'Without-pay', potentially due to fewer instances of these categories in the training data or model limitations in capturing diverse income determinants.

Furthermore, the model shows variability in performance across categories like marital status, race, and sex. For instance, in the 'marital-status' category, the model is more precise in predicting higher income for 'Married-civ-spouse', but less accurate in other categories. Similarly, for 'race' and 'sex', categories such as 'Amer-Indian-Eskimo' and 'Female' exhibit disparities in precision and recall, hinting at potential biases or underrepresentation in the dataset. The varied performance across different occupations also points to a possible alignment of the model's predictions with societal stereotypes or biases present in the training data. This underscores the need for a balanced dataset and a more sophisticated modeling approach to ensure fair and equitable predictions across all socio-demographic groups.

## Ethical Considerations
- The model may reflect biases present in the training data, particularly related to socio-economic factors.
- Caution is advised to prevent adverse impacts on individuals based on demographic or socio-economic status.
- The broader social context and potential reinforcement of existing inequalities should be considered.

## Caveats and Recommendations
- This model is a basic implementation and has room for performance improvement; it is not intended for use in production environments.
- Further work is recommended to improve the model's recall and address any biases in the training data.
