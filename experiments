### Several basic function that helps conducting experiments for logistic regression construction

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import train_test_split
 
plt.rcParams["font.sans-serif"]=["Songti Sc"]
plt.rcParams["axes.unicode_minus"]=False


'''
Input: X, y, model, threshold (defualt=0.5), alpha (default = 0.05)
Output: Metrics with confidence interval
Metrics containing: Accuracy, AUC, precision, negative predictive value (npv), recall (sensitivity), specificity, lrp, lrn, f1score
'''
#==================================================================================================================================
# Compute auc
def compute_auc(gt, pred_prob, pred, alpha):
    auc = round(roc_auc_score(gt, pred_prob), 4)
    n1 = sum(pred==1)
    n2 = sum(pred==0)
    q0 = auc*(1-auc)
    q1 = auc/(2-auc)-auc**2
    q2 = 2*auc**2/(1+auc)-auc**2
    se = np.sqrt((q0+(n1-1)*q1+(n2-1)*q2)/(n1*n2))

    return auc, se

def append_metric(metric_name, metric, metric_list, value, metric_std, z, mode="exp"):
    ci_half = metric_std*z

    if mode == "exp":
        lower_bound = np.exp(np.log(value)-ci_half)
        upper_bound = np.exp(np.log(value)+ci_half)
    else:
        lower_bound = value-ci_half
        upper_bound = value+ci_half

    metric[metric_name] = [value, lower_bound, upper_bound]
    metric_list.append(f"{value:.4f} ({lower_bound:.4f} - {upper_bound:.4f})")
    return metric, metric_list

def metric_analysis(X, y, model, thre=0.5, alpha=0.05, output="df"):
    metric = {}
    metric_list = []

    # two-sided
    z = st.norm.ppf(1-alpha/2)

    gt = y
    pred_prob = model.predict_proba(X)[:,1]
    pred = (pred_prob > thre)+0
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()

    ### accuracy
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    metric["accuracy"] = [accuracy, np.nan, np.nan]
    metric_list.append(accuracy)

    ### auc
    auc, std_auc = compute_auc(gt=gt, pred_prob=pred_prob, pred=pred, alpha=alpha)

    ### precision/positive predictive value
    precision = tp/(tp+fp)
    std_precision = np.sqrt(precision*(1-precision)/(tp+fp))

    ### recall/sensitivity/true positivie rate
    recall = tp/(tp+fn)
    std_recall = np.sqrt(recall*(1-recall)/(tp+fn))

    ### specificity/true negative rate
    specificity = tn/(tn+fp)
    std_specificity = np.sqrt(specificity*(1-specificity)/(tn+fp))

    ### Likelihood ratio positive
    lrp = recall/(1-specificity)
    std_lrp = np.sqrt(1/tp-1/(tp+fn)+1/fp-1/(fp+tn))

    ### Likelihood ratio negative
    lrn = (1-recall)/specificity
    std_lrn = np.sqrt(1/fn-1/(tp+fn)+1/tn-1/(fp+tn))

    metric_names = ["AUC", "Precision/Positive Predictive Value", "Recall/Sensitivity/True Positive Rate", "Specificity/True Negative Rate",
                    "Likelihood Ratio Positive", "Likelihood Ratio Negative"]
    metric_value = [auc, precision, recall, specificity, lrp, lrn]
    metric_std = [std_auc, std_precision, std_recall, std_specificity, std_lrp, std_lrn]
    modes = ["normal", "normal", "normal", "normal",  "exp", "exp"]

    for i in range(len(modes)):
        metric, metric_list = append_metric(metric_names[i], metric, metric_list, metric_value[i], metric_std[i], z=z, mode=modes[i])

    ### F1 score
    f1 = tp/(tp+(fp+fn)/2)
    metric["F1 Score"] = [f1, np.nan, np.nan]
    metric_list.append(f"{f1:.4f}")

    metric = pd.DataFrame(metric).T.reset_index(drop=False)
    metric.columns = [f"Metric (alpha={alpha})", "Value", 'Lower bound', "Upper bound"]

    if output == "df":
        return metric
    if output == "list":
        return metric_list

#==================================================================================================================================
### plot confusion_matrix
def confusion_matrix_plot(X, y, model, title="Confusion Matrix", thre=0.5, normalize=None):
    gt = np.array(y, dtype=int)
    pred_prob = model.predict_proba(X)[:,1]
    pred = (pred_prob > thre)+0
    pred = np.array(pred, dtype=int)
    cm = confusion_matrix(gt, pred, normalize=normalize)

    fig = plt.figure(figsize=(4,4), dpi=80)
    plt.imshow(cm, cmap='Blues', alpha=0.7)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0,1],[0, 1])
    plt.yticks([0,1],[0, 1])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize == None:
                text = plt.text(j, i, '{:d}'.format(cm[i, j]), ha="center", va="center", color="k")
            else:
                text = plt.text(j, i, '{:.4f}'.format(cm[i, j]), ha="center", va="center", color="k")
    return fig


### plot auc curve
def roc_auc_plot(X, y, model, thre=0.5):
    gt = y
    pred_prob = model.predict_proba(X)[:,1]

    fig = plt.figure(figsize=(5,4),dpi=80)
    fpr, tpr, _ = roc_curve(gt, pred_prob)
    roc_score = roc_auc_score(gt, pred_prob)
    plt.plot(fpr, tpr, color='tab:red', lw=2)
    plt.plot([0,1],[0,1], color='tab:gray', linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('AUC: {:.4f}'.format(roc_score), fontsize=18)
    plt.tight_layout()

    return fig


### plot violin plot
def violin_plot(X, y, model, title="Violin Plot", thre=0.5):
    gt = y
    pred_prob = model.predict_proba(X)[:,1]

    fig, ax = plt.subplots(figsize=(5,4),dpi=80)
    ax.violinplot([pred_prob[gt==0], pred_prob[gt==1]], widths=0.8)
    ax.set_ylabel('Probabilities', fontsize=15)
    plt.xticks(np.arange(3), ['',"Negative", "Positive"] )
    plt.tight_layout()

    return fig


def plot_all(X, y, model, thre=0.5):
    fig_confusion = confusion_matrix_plot(X, y, model, thre=thre)
    fig_roc = roc_auc_plot(X, y, model, thre=thre)
    fig_violin = violin_plot(X, y, model, thre=thre)
    return fig_confusion, fig_roc, fig_violin

#==================================================================================================================================

### find the optimal threshold with highest g-mean among validation cohorts
def optimal_cutoff(gt, pred_prob):
    fpr, tpr, threshold = roc_curve(gt, pred_prob)
    gmeans = np.sqrt(tpr*(1-fpr)) ### you can chage your criterion here
    ix = np.argmax(gmeans)
    return threshold[ix]

'''
Input: X, y, test sample proportion, logistic regression / random forest parameters
Output: Best threshold, Best model, result_dataframe
Metrics containing: Accuracy, AUC, precision, negative predictive value (npv), recall (sensitivity), specificity, lrp, lrn, f1score
'''
class myLrRf():
    def __init__(self, X, y, test_size, lr_para={}):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.lr_para = lr_para

    ### run experiments with multiple seeds
    ### and return the model class
    def experiment_multi(self, random_states, metric_sort="AUC", alpha=0.05, ml_model="lr"):
        result = {}
        n = ["train", "val"]
        metric_name = ["Accuracy", "AUC", "Precision", "Sensitivity", "Specificity", "Likelihood Ratio Positive", "Likelihood Ratio Negative", "F1 Score"]
        colname = ["random_seed"]
        for ns in n:
            for name in metric_name:
                colname.append(f"{name}_{ns}")
        
        for random_state in random_states:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=random_state)
            
            X_list = [X_train, X_test]
            y_list = [y_train, y_test]
            
            if ml_model == "lr":
                model = LogisticRegression(**self.lr_para).fit(X_train, y_train)
            if ml_model == "rf":
                model = RandomForestClassifier(**self.lr_para).fit(X_train, y_train)
            threshold = optimal_cutoff(y_train, model.predict_proba(X_train)[:, 1])

            r = []
            for i in range(0,2):
                r += metric_analysis(X_list[i], y_list[i], model, thre=threshold, alpha=alpha, output="list")
            
            result[random_state] = r
        
        result = pd.DataFrame(result).T.reset_index()
        result.columns = colname

        self.multi_result = result

        if metric_sort == None:
            result_for_analysis = result.copy()
        else:
            result_for_analysis = result.sort_values(by = f"{metric_sort}_val", ascending=False).reset_index(drop=True)

            self.result_sorted = result_for_analysis
            self.random_state_best = result_for_analysis.loc[0, "random_seed"]
            self.X_train_best, self.X_test_best, self.y_train_best, self.y_test_best = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state_best)

            if ml_model == "lr":
                self.model_best = LogisticRegression(**self.lr_para).fit(self.X_train_best, self.y_train_best)
                self.coefficient_best = self.model_best.coef_[0]
                self.intercept_best = self.model_best.intercept_
            if ml_model == "rf":
                self.model_best = RandomForestClassifier(**self.lr_para).fit(self.X_train_best, self.y_train_best)
                self.feature_importances_best = self.model_best.feature_importances_

            self.train_proba_best = self.model_best.predict_proba(self.X_train_best)[:,1]
            self.test_proba_best = self.model_best.predict_proba(self.X_test_best)[:,1]
            self.threshold_best = optimal_cutoff(self.y_train_best, self.model_best.predict_proba(self.X_train_best)[:, 1])
            self.metric_best = self.result_sorted.loc[0,:]

    ### run single experiment with specific random seed
    def experiment_single(self, random_state, alpha=0.05, ml_model="lr"):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=random_state)

        if ml_model == "lr":
            self.model = LogisticRegression(**self.lr_para).fit(self.X_train, self.y_train)
            self.coefficient = self.model.coef_[0]
            self.intercept = self.model.intercept_
        if ml_model == "rf":
            self.model = RandomForestClassifier(**self.lr_para).fit(self.X_train, self.y_train)
            self.feature_importances = self.model.feature_importances_
            
        self.train_proba = self.model.predict_proba(self.X_train)[:,1]
        self.test_proba = self.model.predict_proba(self.X_test)[:,1]
        self.threshold = optimal_cutoff(self.y_train, self.model.predict_proba(self.X_train)[:, 1])
        self.metric = metric_analysis(self.X_test, self.y_test, self.model, thre=self.threshold, alpha=alpha, output="df")
        self.fig_confusion_train, self.fig_roc_train, self.fig_violin_train = plot_all(self.X_train, self.y_train, self.model, self.threshold)
        self.fig_confusion_test, self.fig_roc_test, self.fig_violin_test = plot_all(self.X_test, self.y_test, self.model, self.threshold)
