# Dependencies
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_curve, f1_score, precision_recall_curve, roc_auc_score, confusion_matrix)
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas_flavor as pf

@pf.register_dataframe_method
def rocauc_pr_curve_plot(model, 
                        X_t,
                        y_t,
                        label              = "Model",
                        curves_color       = '#0E4978',
                        curves_linewidth   = 1,
                        no_skill_color     = "black",
                        no_skill_linewidth = 1,
                        star_color         = '#FFB81C',
                        star_size          = 200,
                        grid_color         = '#DDDDDD',
                        grid_linewidth     = 0.5,
                        title_weight       = 'bold',
                        title_size         = 12,
                        ax_title_weight    = 'bold',
                        ax_title_size      = 10,
                        left_tick          = True,
                        bottom_tick        = True,
                        legend_size        = 'small',
                        figsize            = (16, 16)
                    ):
    """
    Generates an ROC-AUC Curve with Youden's J-Statistic, which is the geometric mean of the curve, and also represents 
    the precise threshold for optimal balance between false positive and true positive rates. As well as a Precision-Recall
    curve and the F1-Score, which is the harmonic mean of precision and recall, and also represents the precise threshold 
    that provides the optimal balance between precision and recall.
    
    Args: 
      model ([model object]): 
        Model object generated from sklearn or xgboost, e.g., xgboost.sklearn.XGBClassifier.
        
      X_t ([pandas.DataFrame]): 
          X_train, X_valid, or X_test data.
        
      y_t ([pandas.Series]): 
          y_train, y_valid, or y_test data.
        
      label ([str, optional]):
          Label for the ROC-AUC Curve and Precision-Recall Curve that is generated from the model. Defaults to 'Model'.
      
      curves_color ([str, optional]):      
          Color for the ROC-AUC Curve and Precision-Recall Curve. Defaults to Blue, '#0E4978'.
      
      curves_linewidth ([float, optional]):
          The width of the curves. Defaults to 1.
      
      no_skill_color     
          Color of the 'no skill' lines. Defaults to "black",
      
      no_skill_linewidth ([float, optional]):
          The width of the 'no skill' lines. Defaults to 1.
      
      star_color         
          Color of the star that denotes Youden's J-statistic and the F1-Score. Defaults to Gold, '#FFB81C'.
      
      star_size ([float, optional]):          
          Size of the star marker. Defaults to 200.
      
      grid_color       
          Color of the grid. Defaults to '#DDDDDD'.
      
      grid_linewidth ([float, optional]):  
          Line width of the grid. Defaults to 0.5.
      
      title_weight ([str, optional]):      
          Weight of the title font. Defaults to 'bold'.
      
      title_size ([float, optional]):       
          Size of the title font. Defaults to 12.
      
      ax_title_weight ([str, optional]): 
          Weight of the axis title fonts. Defaults to 'bold'.
      
      ax_title_size ([float, optional]):    
          Size of the axis title fonts. Defaults to 10.
      
      left_tick ([boolean, optional]):        
          Boolean indicating to plot y-axis tick marks. Defaults to True.
      
      bottom_tick ([boolean, optional]):      
          Boolean indicating to plot x-axis tick marks. Defaults to True.
      
      legend_size ([str or int, optional]):        
          String {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'} or integer for the legend size. Defaults to 'small'.
      
      figsize ([int, int, optional]):         
          Integer values for the figure size. Defaults to (16, 16).
    
    Returns:
        [matplotlib.axes._subplots.AxesSubplot] of the ROC-AUC Curve and the Precision-Recall Curve.
        
    
    Example:
    
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000,n_features=4,n_informative=3, n_redundant=1, random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
    >>> model = LogisticRegression(solver='lbfgs')
    >>> model.fit(X_train, y_train)
    >>> # Plot Curves
    >>> rocauc_pr_curve_plot(model, X_train, y_train)
    
    """

    # Figure Size for ROC-AUC, Precision Recall, Gain, and Lift Plots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Classify y_pred based upon threshold
    y_pred = model.predict_proba(X_t)
    y_pred = y_pred[:,1]
    fpr, tpr, thresholds = roc_curve(y_t, y_pred)
    youdens_j = tpr - fpr
    ix = np.argmax(youdens_j)
    optimal_ROCAUC_threshold = thresholds[ix]

    # Plot ROCAUC
    plt.subplot(2,2,1)
    plt.plot(fpr, tpr, zorder=1, label=label, color=curves_color, linewidth=curves_linewidth)
    plt.plot([0,1], [0,1], zorder=2, linestyle='--', label='No Skill', color=no_skill_color, linewidth=no_skill_linewidth)
    plt.scatter(fpr[ix], tpr[ix], zorder=3, marker='*', s = star_size, label="Youden's-J="+str(round(youdens_j[ix],4)), color=star_color)
    plt.scatter(fpr[ix], tpr[ix], zorder=3, marker='', s = star_size, label="Threshold="+str(round(thresholds[ix],4)), color=star_color)
    plt.title("ROC AUC Curve", fontweight=title_weight, fontsize=title_size)
    plt.xlabel("False Positive Rate", fontweight=ax_title_weight, fontsize=ax_title_size)
    plt.ylabel("True Positive Rate", fontweight=ax_title_weight, fontsize=ax_title_size)
    plt.grid(which='major', color=grid_color, linewidth=grid_linewidth)
    plt.tick_params(left=left_tick, bottom=bottom_tick)
    plt.legend(loc=0, fontsize=legend_size)

    # Classify y_pred based upon threshold
    y_pred = model.predict_proba(X_t)
    y_pred = y_pred[:,1]
    precision, recall, thresholds = precision_recall_curve(y_true=y_train , probas_pred=y_pred)
    # convert to f score
    fscore = 2 * (precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    # Best Threshold
    optimal_precision_recall_threshold = thresholds[ix]
    # No Skill Line is the Proportion of the Positive class
    no_skill = len(y_train[y_t==1]) / len(y_t)

    # Plot Precision Recall Curve
    plt.subplot(2,2,2)
    plt.plot(recall, precision,  label=label, color=curves_color, linewidth=curves_linewidth)
    plt.plot([0,1], [no_skill, no_skill], linestyle='--', label='No Skill', color = no_skill_color, linewidth=no_skill_linewidth)
    plt.scatter(recall[ix], precision[ix], zorder=3, marker='*', s = star_size, label="Max F1-Score="+str(round(fscore[ix],4)), color=star_color)
    plt.scatter(recall[ix], precision[ix], zorder=3, marker='', s = star_size, label="Threshold="+str(round(thresholds[ix],4)), color=star_color)
    plt.title("Precision Recall Curve", fontweight=title_weight, fontsize=title_size)
    plt.xlabel("Recall", fontweight=ax_title_weight, fontsize=ax_title_size)
    plt.ylabel("Precision", fontweight=ax_title_weight, fontsize=ax_title_size)
    plt.grid(which='major', color=grid_color, linewidth=grid_linewidth)
    plt.tick_params(left=left_tick, bottom=bottom_tick)
    plt.legend(loc=0, fontsize=legend_size)
    
    return
