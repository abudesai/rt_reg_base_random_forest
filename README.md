Random Forest Regressor using Scikit-Learn for Regression

* random forest 
* bagging
* ensemble
* python
* feature engine
* scikit optimize
* flask
* nginx
* gunicorn
* docker
* abalone
* auto prices
* computer activity
* heart disease
* white wine quality
* ailerons

This is an implementation of Random forest using Scikit-Learn. 

A Random Forest algorithm fits a number of decision trees on various samples of the dataset and uses mean of all outputs to improve the predictive accuracy and controls over-fitting. 

The sample size is controlled by the max_samples parameter and bootstrapping is generally used as default, otherwise the entire dataset is used to build each tree. 


Preprocessing includes missing data imputation, standardization, one-hot encoding etc. For numerical variables, missing values are imputed with the mean and a binary column is added to represent 'missing' flag for missing values. For categorical variable missing values are handled using two ways: when missing values are frequent, impute them with 'missing' label and when missing values are rare, impute them with the most frequent. 

HPT based on Bayesian optimization is included for tuning Random Forest hyper-parameters. 


The main programming language is Python. Other tools include Scikit-Learn for main algorithm, feature-engine for preprocessing, Scikit-Optimize for HPT, Flask + Nginx + gunicorn for web service. The web service provides two endpoints- /ping for health check and /infer for predictions in real time.