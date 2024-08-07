# Iowa Liquor Sales Forecast

Please explore the following datasets, there are description and data
dictionaries available in the links to Iowa liquor sales:

- Publicly available on
  `BigQuery`: https://console.cloud.google.com/projectselector2/bigquery?p=bigquery-public-data&d=iowa_liquor_sale
- Same datasets are also available here for more
  detail: https://data.iowa.gov/Sales-Distribution/Iowa-Liquor-Sales/m3tr-qhgy/data

## Instructions (scripts should be done in Python):

Take the Iowa Liquor Retail Sales dataset from `BigQuery` and build a simple
predictive model for sales forecast. Create functions to run
feature engineering, feature selection. Also create scripts for hyperparameter
tuning, model evaluation, show us the accuracy of the model,
and deploy the model to cloud run (or similar cloud services)
and walk through the code and process of how you set it up.
The model should have a scheduler so that it runs once every week.

**Bonus point:** implement CI/CD

## Minimum quality standards:

- **Good enough:** Builds a reasonable model and can have it executed on the
  cloud.
- **Better:** Same as above. Plus performs better modeling tasks,
  such as more **advanced feature engineering**.
- **Best:** Same as above. Plus implements **CI/CD**, and/or utilizes 
  infrastructure as code such as [Terraform](https://registry.terraform.io/)

The team can use code repo and cloud platform of their choice (but ideally if
they can already try this with GCP would be better).

During the coding sessions, we will use the time to go through the result.
Considering the time given, we are not looking for a perfect solution, just
enough to better understanding of the team’s technical profile and also share
our expectations more clearly.