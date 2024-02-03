# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This Logistic Regression model determines whether a subject's salary is higher than or equal to 50,000 USD or less than that using several imputs such as `Education` and `Employment`.

## Intended Use

This model can be used to predict the salaries of individuals at a specific point in time. Specifically, the census data used to construct the model is from 2014. Therefore, this model can accept input data, when that data's attributes were sampled at 2014.

## Training Data
 
census.gov has made available an anonymized data set of user attributes and salaries. The training data was obtained through an arbitrary horizontal slice of this data.

## Evaluation Data

census.gov has made available an anonymized data set of user attributes and salaries. The evaluation data was obtained through an arbitrary horizontal slice of this data.

## Metrics

The metrics for the model's performance can be found in `data/slice_output.txt`. These metrics show the model's performance, when the model is applied to a dataset that is obtained by restricting the value of the `education` feature to a single value. For example, there is a metric that measures model performance when the data set solely consists of persons whose `education` is `Bachelors`. Likewise, there is a metric that measures performance when `education` is `Doctorate`, and so forth.

The metrics used are `precision`, `recall` and `fbeta`.

## Ethical Considerations

Please don't use this on contemporary humans.

## Caveats and Recommendations

One hot encoder is used to transform categorical features and their values. Please bear this in mind, because if you wish to use the model, then you will need to encode your data using the encoder that has also been disclosed in this repository.