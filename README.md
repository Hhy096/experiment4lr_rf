## Experiment4lr_rf

Design an integrated package for running experiments of logistic regression or random forest models with different train and validation random seeds (Binary problem), return the model and metrics in both train and validation cohorts.

>Functions Illustration
  - **compute_auc**(gt, pred_prob, pred, alpha):  
    **Return the AUC and corresponding confidence interval for given ground truth and prediction probability.**
      - **gt:** the groud truth label.
      - **pred_prob:** the prediction probability returned by the model.
      - **pred:** the prediction outcome (binary) returned by the model.
      - **alpha:** alpha for confidence interval, usally alpha=0.05.  
    &nbsp;
  - **metric_analysis**(X, y, model, thre=0.5, alpha=0.05, output="df"):
    **Return the metrics (accuracy, auc, precision, recall, specificity, likelihood ratio ositive, likelihood ratio negative) dataframe or list of the machine learning model.**
      - **X:** the dataset you want to test the model in.
      - **y:** the corresponding labels of dataset **X**.
      - **model:** the model required to test.
      - **thre:** the threshold of the model for distinguishing positive and negative cases, default 0.5.
      - **alpha:** alpha for confidence interval, default 0.05.
      - **output:** output form of the metric ("df" or "list"), default "df".  
    &nbsp;
  - **confusion_matrix_plot**(X, y, model, thre=0.5):
    **Plot the confusion matrix of the model on specific dataset. Return the curve**
      - **X:** the dataset you want to draw the confusion matrix on.
      - **y:** the corresponding labels of dataset **X**.
      - **model:** the machine learning model required.  
    &nbsp;
  - **roc_auc_plot**(X, y, model, thre=0.5):
    **Plot the ROC curve of the model on specific dataset. Return the figure.**
      - **X:** the dataset you want to draw the ROC curve on.
      - **y:** the corresponding labels of dataset **X**.
      - **model:** the machine learning model required.  
    &nbsp;
  - **violin_plot**(X, y, model, thre=0.5):
    **Plot the violin plots of the predictive probability distributions of different label populations model on specific dataset. Return the figure.**
      - **X:** the dataset you want to draw the ROC curve on.
      - **y:** the corresponding labels of dataset **X**.
      - **model:** the machine learning model required.  

>Class and corresponding methods
  - **myLrRf**(X, y, test_size, lr_para={}):
  **Construct class for further conducting experiments.**
      - **X:** the dataset you want to conduct the experiments on.
      - **y:** the corresponding labels of dataset **X**.
      - **test_size:** the proportion of test cohorts, such as 0.3.
      - **lr_para:** the parameters for logistic regression and random forest.  
  &nbsp;
  - **experiment_multi**(random_states, metric_sort="AUC", alpha=0.05, ml_model="lr"):
  **Construct class for further conducting experiments.**
      - **random_states:** list of random seeds the experiment will iterate on.
      - **metric_sort:** the metric the result dataframe is sorted by, default "AUC". The values can be selected from ("Accuracy", "AUC", "Precision", "Sensitivity", "Specificity", "Likelihood Ratio Positive", "Likelihood Ratio Negative", "F1 Score").
      - **alpha:** alpha for confidence interval, default 0.05.
      - **ml_model:** the chosen machine learning model, default "lr". The values can be selected from ("lr", "rf").  
  &nbsp;
  - **experiment_single**(random_state, alpha=0.05, ml_model="lr"):
  **Construct class for further conducting experiments.**
      - **random_state:** the random seed the experiment will conduct on.
      - **alpha:** alpha for confidence interval, default 0.05.
      - **ml_model:** the chosen machine learning model, default "lr". The values can be selected from ("lr", "rf").  

>mtLrRf class and its methods
  - **self.X:** the whole datasets containing the input variables.
  - **self.y:** the whole datasets containing the labels.
  - **self.test_size:** the test cohort proportion.
  - **self.lr_para:** the required parameters for logistic regression of random forest.
  - **self.result_sorted:** the metric dataframe sorted by the specific metric.
  - **self.model_best:** the best model with the highest specific metric.
  - **self.model_coefficient_best:** the coefficient of the best model.
  - **self.model_intercept_best:** the intercept of the best model.
  - **self.feature_importances_best:** the feature importances returned by the best random forest model.
  - **self.train_proba_best:** the predictive probability returned by the best model of training set.
  - **self.test_proba_best:** the predictive probability returned by the best model of test set.
  - **self.threshold_best:** the threshold of the best model.
  - **self.metric_best:** the metric for the best model.
  - **self.model:** the model returned by method experiment_single.
  - **self.coefficient:** the coefficient of the model.
  - **self.intercept:** the intercept of the model.
  - **self.feature_importances:** the feature importances returned by the random forest model
  - **self.train_proba:** the predictive probability returned by the model of training set.
  - **self.test_proba:** the predictive probability returned by the model of test set.
  - **self.threshold:** the threshold of the model.
  - **self.metric:** the metric for the model.
  - **self.fig_confusion_train:** the confusion matrix in training data.
  - **self.fig_roc_train:** the ROC curve in training data.
  - **self.fig_violin_train:** the violin plot in training data.
  - **self.fig_confusion_test:** the confusion matrix in test data.
  - **self.fig_roc_test:** the ROC curve in test data.
  - **self.fig_violin_test:** the violin plot in testa data.

### For specific examples, please refer to 
