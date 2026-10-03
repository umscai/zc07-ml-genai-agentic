"""
This file contains the utility functions used in this project
"""

import os
import math
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike # for type-hint
import pandas as pd
import seaborn as sns
from seaborn.axisgrid import JointGrid # for type-hint
import matplotlib.pyplot as plt
from matplotlib.axes import Axes # for type-hint
from typing import Optional, List, Dict, Any, Tuple
from config import settings
import requests
import io
import zipfile
import kagglehub
import shutil
import json
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    fbeta_score,
    make_scorer,
)
import joblib
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm  # Auto-adapts to Notebook or Terminal environment
from matplotlib.colors import LogNorm
import logging
logger = logging.getLogger(__name__)


#----------------------------------------------------------------------------------------
def download_dataset_from_kagglehub(dataset_id:str = settings.DATASET_ID,
                                    dataset_path:str = settings.DATASET_PATH,
                                    dl_dir:Path = settings.DATA_DIR):
    """
    download dataset from kagglehub into standard kagglehub-cache
    then copy to local dl_dir, if not downloaded yet
    Args:
        dataset_id: dataset id
        dataset_path: dataset file-path or a representative when we have multiple files
        dl_dir: destination download directory
    Returns:
        None.
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    if dataset_path.exists():
        print(f"dataset is already downloaded in {dl_dir}")
        return
    try:
        cache_path = kagglehub.dataset_download(dataset_id)
        print(f'dataset downloaded in kagglehub-cache-path = {cache_path}')

        shutil.copytree(cache_path, dl_dir, dirs_exist_ok=True)
        print(f"Download and copy completed! Files saved in {dl_dir} folder.")

    except Exception as e:
        logger.error(f"Error in download_dataset_from_kagglehub: {e}")
        raise

#----------------------------------------------------------------------------------------
def check_duplicate(df: pd.DataFrame):
    """
    check and show total number of duplicated rows
     Args:
        df: Input dataframe
    Returns:
        None.
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        duplicate_count = df.duplicated().sum()
        duplicate_count_pct = 100*duplicate_count/len(df)
        print(f"Total duplicate rows: {duplicate_count} ({duplicate_count_pct:.1f}%)")
    except Exception as e:
        logger.error(f"Error in check_duplicate: {e}")
        raise

#----------------------------------------------------------------------------------------
def remove_duplicate(df: pd.DataFrame)-> pd.DataFrame:
    """
    remove duplicated rows and return resulting dataframe
     Args:
        df: Input dataframe
    Returns:
        Dataframe without duplicate.
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        L1 = len(df)
        df = df.drop_duplicates().reset_index(drop=True).copy()
        L2 = len(df)
        print(f'number of rows Before and After removing duplicates: Before = {L1},  After = {L2}')
        return df
    except Exception as e:
        logger.error(f"Error in remove_duplicate: {e}")
        raise

#----------------------------------------------------------------------------------------
def check_nan_values(df: pd.DataFrame)->pd.DataFrame|None:
    """
    check and doe NaN (missing) values
    Args:
        df: Input dataframe
    Returns:
        missing col/count/percentage in dataframe format when we have missing otherwise None
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        total_missing_count = df.isnull().sum().sum()
        if total_missing_count == 0:
            #print('+' * 80)
            print('No missing (NaN) values in dataset')
            print('+' * 80)
            return None
        else:
            #print('+' * 80)
            print(f'There are total {total_missing_count} missing (NaN) values in dataset')
            print('+' * 80)
            missing_count_df = df.isnull().sum().to_frame(name='missing_count').reset_index(names=['column_name'])
            missing_count_df = missing_count_df[missing_count_df['missing_count']>0]
            missing_count_df['missing_count_pct']=(100*missing_count_df['missing_count']/len(df)).round(1)
            missing_count_df = missing_count_df.sort_values(by='missing_count_pct', ascending=False)
            return missing_count_df #display(missing_count_df)
    except Exception as e:
        logger.error(f"Error in check_nan_values: {e}")
        raise

#----------------------------------------------------------------------------------------
def show_bar_plot(df: pd.DataFrame, col: str,
                  sns_palette:str|List[str]='viridis_r', top_or_bottom_n:int = 0, ext_ax:Optional[Axes] = None, ext_str:str='', rot=False):
    """
    Show bar plot of a given categorical/ordinal column in a dataframe, with the option to just show top or bottom n
    Args:
        df: Input dataframe
        col: name of targeted categorical column
        sns_palette: seaborn palette name (e.g. viridis_r, crest_r,  flare_r, magma, Blues_r,...)
        top_or_bottom_n: determine how many of top or bottom list to be shown on plot (0: all, positive=top, negative = bottom)
        ext_ax: external existing matplotlib axis for plot
        ext_str: extra string for title and label
        rot: force x-label rotation if True
    Returns:
        None
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        if ext_ax is None:
            fig, ext_ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

        col_counts = df[col].value_counts().reset_index()

        if top_or_bottom_n > 0:
            col_counts_ = col_counts.head(top_or_bottom_n)
            title_ext = f' (Top {top_or_bottom_n})'
        elif top_or_bottom_n < 0:
            col_counts_ = col_counts.tail(-top_or_bottom_n)
            title_ext = f' (Bottom {-top_or_bottom_n})'
        else:
            col_counts_ = col_counts
            title_ext = ''

        sns.barplot(data=col_counts_, x=col, y='count', dodge=False,
                    hue=col, palette=sns_palette, legend=False, ax=ext_ax)

        total_counts = len(df) # same as col_counts['count'].sum()
        if len(col_counts_) < 20: # add % label only if we have less than 20 bars
            for patch in ext_ax.patches:
                height = patch.get_height()
                if height > 0:
                    percentage = f"{(height / total_counts) * 100:.1f}%"
                    x = patch.get_x() + patch.get_width() / 2
                    y = height + 0.1 #(total_counts * 0.01)
                    ext_ax.annotate(percentage, (x, y), ha="center", va="bottom", fontsize=10, fontweight="bold")

        ext_ax.set_title(f'Number of Records From Each {col}{title_ext}{ext_str}', fontsize=12)
        ext_ax.set_xlabel(f'{col}{ext_str}')
        ext_ax.set_ylabel(f'Count of Records{ext_str}')

        #plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ext_ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        #plt.gca().set_axisbelow(True) # make sure bar on top of grid
        ext_ax.set_axisbelow(True)
        # if top_or_bottom_n !=0 or rot:
        #     #plt.xticks(rotation=45, ha='right', fontsize=11)
        #     ext_ax.set_xticklabels(ext_ax.get_xticks(), rotation=45, ha='right', fontsize=11)
        if top_or_bottom_n != 0 or rot:
            # Option 1: Using tick_params (Cleanest and recommended)
            ext_ax.tick_params(axis='x', rotation=45, labelsize=11)

            # Note: ha='right' might require setting it on the text objects directly if using tick_params:
            for label in ext_ax.get_xticklabels():
                label.set_ha('right')

        if len(col_counts_) >= 30: #drop all xtick-label when we have more than 20 bar in this project
            ext_ax.set_xticklabels([])

    except Exception as e:
        logger.error(f"Error in show_bar_plot for column '{col}': {e}")
        raise

#----------------------------------------------------------------------------------------
def show_box_plot_for_cols(df: pd.DataFrame, cols:List[str], n_cols_grid: int=2, sns_palette:str='tab20', show_1_99_pct:bool=False):
    """
    Show box-plot for given features in a dataframe, in a grid of subplot with given number of columns,
    and use color from given sns_palette for different subplot/box-plot
    Args:
        df: Input dataframe
        cols: list of columns/features to be plotted
        n_cols_grid: number of columns in the subplots configuration
        sns_palette: seaborn palette name (e.g. viridis_r, crest_r,  flare_r, magma, Blues_r, tab20,...)
        show_1_99_pct: if True we also calculate 1th and 99th percentile and show it on same boxplot
    Returns:
        None
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        # Set up grid layout
        n_rows_grid = math.ceil(len(cols) / n_cols_grid)

        fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(18, 4 * n_rows_grid))
        axes_flat = axes.flatten() if n_rows_grid > 1 or n_cols_grid > 1 else [axes]

        # 2. Loop through each column and plot on its own dedicated axis with a unique color
        palette = sns.color_palette("tab20", len(cols))

        for i, col in enumerate(cols):
            ax = axes_flat[i]
            sns.boxplot(data=df, y=col, ax=ax, color=palette[i])

            ax.set_title(f'Boxplot: {col}', fontsize=12)
            ax.set_ylabel(col, fontsize=10)
            ax.grid(True, alpha=0.3)
            if show_1_99_pct:
                p01 = df[col].quantile(0.01)
                p99 = df[col].quantile(0.99)
                ax.axhline( y=p99, color='red', linestyle='--', linewidth=1.5, label=f'99th Pct')
                ax.axhline( y=p01, color='orange', linestyle='-.', linewidth=1.5, label=f'01th Pct')
                ax.legend(loc='upper right', fontsize=8)

        # Hide any unused trailing subplots
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
    except Exception as e:
        logger.error(f"Error in show_box_plot_for_numerical_data: {e}")
        raise

#----------------------------------------------------------------------------------------
def show_class_dist_on_split_data(train_df: pd.DataFrame, test_df:pd.DataFrame, T:int=1):
    """
    Show 0/1 class distribution on train-test split data
    Args:
        train_df: Input dataframe of train dataset
        test_df: Input dataframe of test dataset
        T: grouping factor on time
    Returns:
        None
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        #H='Hour' if T==1 else f'Hour/{T}'
        H= f'Hour/{T}'
        fig, axs = plt.subplots(2,2,figsize=(12,7))
        colors= {0: "forestgreen", 1: "crimson"}
        sns.histplot(data=train_df[train_df.Class==0], x=H, kde=True, ax=axs[0,0], color=colors[0],bins=24)
        sns.histplot(data=train_df[train_df.Class==1], x=H, kde=True, ax=axs[0,1], color=colors[1],bins=24)
        axs[0,0].set_title(f'Not-Fraud Histogram by {H} (Train dataset)');
        axs[0,1].set_title(f'Fraud Histogram by {H} (Train dataset)');

        colors= {0: "green", 1: "red"}
        sns.histplot(data=test_df[test_df.Class==0], x=H, kde=True, ax=axs[1,0], color=colors[0],bins=24)
        sns.histplot(data=test_df[test_df.Class==1], x=H, kde=True, ax=axs[1,1], color=colors[1],bins=24)
        axs[1,0].set_title(f'Not-Fraud Histogram by {H} (Test dataset)');
        axs[1,1].set_title(f'Fraud Histogram by {H} (Test dataset)');

        for ax in axs.flatten():
            ax.set_xticks(range(0, 24//T, 1))

        plt.tight_layout()
    except Exception as e:
        logger.error(f"Error in time_series_to_supervised: {e}")
        raise

#----------------------------------------------------------------------------------------
def show_confusion_matrix(y_true:ArrayLike, y_pred: ArrayLike, xtick:List[str], ytick:List[str], model_name:str,  threshold:float, sns_cmap:str='Blues', ext_ax:Optional[Axes] = None):
    """
    Show confusion matrix
    Args:
        y_true: true-label values
        y_pred: predicted-label values
        xtick: list of x-tick-label
        ytick: list of y-tick-label
        model_name: name of model used for prediction e.g. GNB or LR,...
        threshold: threshold used for label classification, use for title string
        sns_cmap: cmap used in heatmap that construct confusion matrix
        ext_ax: external existing matplotlib axis for plot
    Returns:
        None
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        if ext_ax is None:
            fig, ext_ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap=sns_cmap, xticklabels=xtick, yticklabels=ytick, ax=ext_ax)
        ext_ax.set_title(f'{model_name}: Confusion Matrix (Thr: {threshold:.3f})', fontsize=11)
        ext_ax.set_xlabel('Predicted Label', fontsize=12)
        ext_ax.set_ylabel('True Label', fontsize=12)
        #plt.tight_layout()
    except Exception as e:
        logger.error(f"Error in show_confusion_matrix: {e}")
        raise

#----------------------------------------------------------------------------------------
def get_model_scores(X: pd.DataFrame, y:pd.Series, model_name:str, model: Any, threshold:Optional[float]=None) -> Dict[str,Any]:
    """
    return all required scores in dict format (metric, value), also include model-name in dict
    Args:
        X: X_train or X_test in pandas data-frame format
        y: y_train or y_test in pandas series format
        model_name: model name for tracking purpose
        model: the model instance like XGBClassifier/CatBoostClassifier/RandomForestClassifier
        threshold: threshold value that we apply on probability to get 0/1 class-label
    Returns:
        dict of score-name, score-value
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        y_probs = model.predict_proba(X)[:, 1] #note: output is [Probability_of_Class_0, Probability_of_Class_1
        if threshold is None:
            y_pred = model.predict(X)
            threshold = 0.5
        else:
            y_pred = (y_probs >= threshold).astype(int) # instead of y_pred = model.predict(X)

        # Calculate metrics
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average="binary", zero_division=0)
        rec = recall_score(y, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y, y_pred, average="binary", zero_division=0)
        f2= fbeta_score(y, y_pred, average='binary', zero_division=0, beta = 2)
        f3= fbeta_score(y, y_pred, average='binary', zero_division=0, beta = 3)
        #ave_pr = average_precision_score(y, y_probs)

        #ROC-AUC, which plots True Positive Rate (Recall) against False Positive Rate (1 - Specificity)
        roc_auc = roc_auc_score(y, y_probs)

        #Area Under the Precision-Recall Curve, scikit-learn recommends average_precision_score,
        # Recommended way to compute PR-AUC
        pr_auc = average_precision_score(y, y_probs)


        row={}
        row = {
            "Model": model_name,
            "Threshold": threshold,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "F2-Score": f2,
            "F3-Score": f3,
            "PR-AUC": pr_auc,
            "ROC-AUC": roc_auc
        }
        #print(confusion_matrix(y, y_pred))
        return row
    except Exception as e:
        logger.error(f"Error in get_model_scores: {e}")
        raise

#----------------------------------------------------------------------------------------
def show_model_auc_comparisons(model_name1:str, model1:Any, model_name2:str, model2:Any, X_test:pd.DataFrame, y_test:pd.Series):
    """
    Plots side-by-side ROC and Precision-Recall curves for two models using X_test and y_test.
    Args:
        X: X_test in pandas data-frame format
        y: y_test in pandas series format
        model_name: model name for tracking purpose
        model: the model instance like XGBClassifier/CatBoostClassifier/RandomForestClassifier
    Returns:
        None
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        colors=['forestgreen','crimson'] #  or blue darkorange
        y_probs1 = model1.predict_proba(X_test)[:, 1]
        y_probs2 = model2.predict_proba(X_test)[:, 1]

        fpr1, tpr1, _ = roc_curve(y_test, y_probs1)
        roc_auc1 = roc_auc_score(y_test, y_probs1)

        fpr2, tpr2, _ = roc_curve(y_test, y_probs2)
        roc_auc2 = roc_auc_score(y_test, y_probs2)

        prec1, rec1, _ = precision_recall_curve(y_test, y_probs1)
        pr_auc1 = auc(rec1, prec1)

        prec2, rec2, _ = precision_recall_curve(y_test, y_probs2)
        pr_auc2 = auc(rec2, prec2)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(fpr1, tpr1, label=f'{model_name1} (AUC = {roc_auc1:.4f})', color=colors[0], lw=2)
        axes[0].plot(fpr2, tpr2, label=f'{model_name2} (AUC = {roc_auc2:.4f})', color=colors[1], lw=2)
        axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess')
        axes[0].set_title('ROC Curves', fontsize=14)
        axes[0].set_xlabel('False Positive Rate', fontsize=12)
        axes[0].set_ylabel('True Positive Rate (Recall)', fontsize=12)
        axes[0].legend(loc='lower right', fontsize=11)
        axes[0].grid(True, linestyle='--', alpha=0.5)

        axes[1].plot(rec1, prec1, label=f'{model_name1} (PR-AUC = {pr_auc1:.4f})', color=colors[0], lw=2)
        axes[1].plot(rec2, prec2, label=f'{model_name2} (PR-AUC = {pr_auc2:.4f})', color=colors[1], lw=2)
        axes[1].set_title('Precision-Recall Curves', fontsize=14)
        axes[1].set_xlabel('Recall', fontsize=12)
        axes[1].set_ylabel('Precision', fontsize=12)
        axes[1].legend(loc='lower left', fontsize=11)
        axes[1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
    except Exception as e:
        logger.error(f"Error in show_model_auc_comparisons: {e}")
        raise

#----------------------------------------------------------------------------------------
def get_xgb_resolved_params(clf:XGBClassifier)->Dict:
    """
    Get main parameter of fitted XGBClassifier
    Args:
        clf: XGBClassifier instance
    Returns:
        dic of selected parameters and their values
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        # Load raw JSON configuration
        config = json.loads(clf.get_booster().save_config())

        # Navigate to internal sections
        gb = config['learner']['gradient_booster']
        tree_param = gb['tree_train_param']
        model_param = gb['gbtree_model_param']

        return {
            'n_estimators': int(model_param.get('num_trees')),
            'learning_rate': float(tree_param.get('learning_rate')),
            'max_depth': int(tree_param.get('max_depth')),
            'subsample': float(tree_param.get('subsample')),
            'colsample_bytree': float(tree_param.get('colsample_bytree')),
            'gamma': float(tree_param.get('gamma')),
            'reg_alpha': float(tree_param.get('alpha')),
            'reg_lambda': float(tree_param.get('lambda')),
            'tree_method': gb['gbtree_train_param'].get('tree_method'),
            'scale_pos_weight': float(config['learner']['objective']['reg_loss_param'].get('scale_pos_weight', 1.0))
        }
    except Exception as e:
        logger.error(f"Error in show_model_auc_comparisons: {e}")
        raise

#----------------------------------------------------------------------------------------
def train_eval(X_train:pd.DataFrame, y_train:pd.Series,
               X_test:pd.DataFrame, y_test:pd.Series,
               tag:str, metrics:List)->Tuple[XGBClassifier, List]:
    """
    Initia a new XGBoost classifier, fit on given training dadat, evaluate on given test data
    return the model and
    Args:
        X: X_test in pandas data-frame format
        y: y_test in pandas series format
        tag: model name for tracking purpose
        metrics: a list contains existing metrics (to be added by new metric and return)
    Returns:
        a tuple, of trained class, and new metrics list by adding new metrics
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        clf=XGBClassifier(
            eval_metric=['aucpr', 'logloss'],
            random_state=settings.RANDOM_SEED,
            verbosity=0)
        clf.fit(X_train, y_train)
        #----------------------------------------

        th_list=np.linspace(0.1,0.9,20)
        tmp=[]
        for th in th_list:
            tmp.append(get_model_scores(X_test, y_test, tag, clf, th))
        tmp=pd.DataFrame(tmp)
        idx_max = tmp['F2-Score'].idxmax()
        thx=tmp.loc[idx_max,'Threshold']
        #----------------

        for th in [thx,0.5]:
            metrics.append(get_model_scores(X_test, y_test, tag, clf, th))
        #metrics_df=pd.DataFrame(metrics)
        #metrics_df
        model_path=settings.MODEL_DIR/f'{tag}_model.pkl'
        joblib.dump(clf, model_path)
        print(f'{tag}-Model saved at {model_path}')
        return clf, metrics

    except Exception as e:
        logger.error(f"Error in train_eval: {e}")
        raise

#----------------------------------------------------------------------------------------
def plot_2d_visualization_for_binary_class_using_tsne(df: pd.DataFrame, max_class_samples:int=5000, ext_ax:Optional[Axes] = None):
    """
    Using t-SNE to visualize a binary class in 2D map, limit max number of samples per class due to heavy processing
    Args:
        - df: datframe contains binary class: Class=0: Majority=Legitimate Transaction, and Class=1: Minority= Fraud Transaction
        - max_class_samples: max number of sample per class
    Returns:
        None
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        L1 = len(df[df['Class'] == 1])
        L0 = len(df[df['Class'] == 0])
        if L1 > max_class_samples:
            frauds = df[df['Class'] == 1].sample(n=max_class_samples, random_state=settings.RANDOM_SEED)
        else:
            frauds = df[df['Class'] == 1]

        if L0 > max_class_samples:
            legits = df[df['Class'] == 0].sample(n=max_class_samples, random_state=settings.RANDOM_SEED)
        else:
            legits = df[df['Class'] == 0]


        df_subset = pd.concat([frauds, legits], axis=0).sample(frac=1, random_state=settings.RANDOM_SEED).reset_index(drop=True)

        #Before standard scaler map Hour/T into sin cos, so even if we scale these two together keep cyclical nature of original data
        hour_col=f'Hour/{settings.HOURS_IN_TIME_GROUP}'
        max_val=23//settings.HOURS_IN_TIME_GROUP
        df_subset['hour_sin'] = np.sin(2 * np.pi * df_subset[hour_col] / (max_val + 1))
        df_subset['hour_cos'] = np.cos(2 * np.pi * df_subset[hour_col] / (max_val + 1))
        # Drop the original linear column
        df_subset = df_subset.drop(columns=[hour_col])


        y = df_subset['Class']
        X = df_subset.drop(['Class'], axis=1)
        X_scaled = StandardScaler().fit_transform(X)

        print('Running t-SNE... ')
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=settings.RANDOM_SEED)
        X_embedded = tsne.fit_transform(X_scaled)

        # 6. Create a DataFrame for plotting
        df_tsne = pd.DataFrame(X_embedded, columns=['Dimension 1', 'Dimension 2'])
        df_tsne['Class'] = y.values

        # Map labels for nicer plot legends
        df_tsne['Class_Label'] = df_tsne['Class'].map({0: 'Legitimate', 1: 'Fraud'})

        # 7. Plotting the 2D visualization
        if ext_ax is None:
            fig, ext_ax = plt.subplots(1,1, figsize=(8,6))


        sns.scatterplot(x='Dimension 1', y='Dimension 2',hue='Class_Label',
            palette={'Legitimate': 'forestgreen', 'Fraud': 'crimson'}, #{'Legitimate': '#1f77b4', 'Fraud': '#d62728'},
            alpha=0.7, s=30, data=df_tsne, ax=ext_ax)

        ext_ax.set_title('t-SNE Visualization of Credit Card Transactions', fontsize=14)
        ext_ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ext_ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ext_ax.legend(title='Transaction Type')
        ext_ax.grid(True, linestyle='--', alpha=0.5)
    except Exception as e:
        logger.error(f"Error in plot_2d_visualization_for_binary_class_using_tsne: {e}")
        raise

#----------------------------------------------------------------------------------------
def compare_two_metrics(df_metrics: pd.DataFrame, metric1:str, metric2:str, at_best_thr:bool, colors:Optional[List[str]]=None, ext_ax:Optional[Axes] = None):
    """
    Compare Two givem metrics based on the data in df_metrics, in a bar-plot
    Args:
        - df_metrics: contain info on models thresold, and binary classification metrics (like recall, precision, f2-score,...)
        - metric1: name of first metric
        - metric1: name of second metric
        - at_best_thr: if we use best threshold (ignore thresold =0.5)
        - colors: optional color-list to be used for 2 bar-plot
        - ext_ax: optional external axes for plotting
    Returns:
        None
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        if at_best_thr:
            df_metric_thr=df_metrics[df_metrics.Threshold != 0.5]
            df_ranking = df_metric_thr[['Model', metric1, metric2]].drop_duplicates().sort_values(by=metric1, ascending=False)
            title_ext=' at best threshold'
        else:
            df_ranking = df_metrics[['Model', metric1, metric2]].drop_duplicates().sort_values(by=metric1, ascending=False)
            title_ext=''
        sns.set_theme(style='whitegrid')
        if ext_ax is None:
            fig, ext_ax = plt.subplots(figsize=(10, 4))

        x = np.arange(len(df_ranking))
        width = 0.35

        if colors is None:
            colors = plt.get_cmap('tab20').colors
        rects1 = ext_ax.bar(x - width/2, df_ranking[metric1], width, label=metric1, color=colors[0])
        rects2 = ext_ax.bar(x + width/2, df_ranking[metric2], width, label=metric2, color=colors[1], alpha=0.7)

        ext_ax.set_title(f'Overall Ranking Performance ({metric1} vs. {metric2}){title_ext}', fontsize=13, fontweight='bold', pad=12)
        ext_ax.set_xticks(x)
        ext_ax.set_xticklabels(df_ranking['Model'], fontweight='bold')
        ext_ax.set_ylim([0.75, 1.02])
        ext_ax.set_ylabel('Score', fontsize=11)
        ext_ax.legend(loc='lower right')

        # Value labels on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ext_ax.annotate(f'{height:.4f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        autolabel(rects1)
        autolabel(rects2)
    except Exception as e:
        logger.error(f"Error in compare_two_metrics: {e}")
        raise

#----------------------------------------------------------------------------------------
def show_all_model_auc_comparisons(model_dic:Dict[str, Any], X_test:pd.DataFrame, y_test:pd.Series, models_name_in_order:Optional[List[str]]=None):
    """
    Plots side-by-side ROC and Precision-Recall curves for All models in model_dic using X_test and y_test.
    Args:
        model_dic: dict contains model name and the model (XGB-classifier)
        X: X_test in pandas data-frame format
        y: y_test in pandas series format
    Returns:
        None
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        fig, axs= plt.subplots(1,2, figsize=(14, 5))

        colors = plt.get_cmap('tab10').colors
        if models_name_in_order is None:
            models_name_in_order = model_dic.keys()

        for j, model_name in enumerate(models_name_in_order):
            model = model_dic[model_name]
            y_proba = model.predict_proba(X_test)[:, 1]

            # Calculate PR curve points & PR-AUC
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
            axs[0].plot(recall, precision, label=f'{model_name} (PR-AUC = {pr_auc:.4f})', color=colors[j], linewidth=2)

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = roc_auc_score(y_test, y_proba)
            axs[1].plot(fpr, tpr, label=f'{model_name} (ROC-AUC = {roc_auc:.4f})', color=colors[j], linewidth=2)


        axs[0].set_title('Precision-Recall Curves across Sampling Strategies', fontsize=13, fontweight='bold', pad=12)
        axs[0].set_xlabel('Recall (Fraud Detection Rate)', fontsize=11)
        axs[0].set_ylabel('Precision (Positive Predictive Value)', fontsize=11)
        axs[0].set_xlim([0.0, 1.0])
        axs[0].set_ylim([0.0, 1.05])
        axs[0].legend(loc='lower left', frameon=True)
        axs[0].grid(True, linestyle='--', alpha=0.5)


        axs[1].set_title('FPR-TPR Curves across Sampling Strategies', fontsize=13, fontweight='bold', pad=12)
        axs[1].set_xlabel('False Positive Rate', fontsize=11)
        axs[1].set_ylabel('True Positive Rate (Recall)', fontsize=11)
        axs[1].set_xlim([0.0, 1.0])
        axs[1].set_ylim([0.0, 1.05])
        axs[1].legend(loc='lower right', frameon=True)
        axs[1].grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
    except Exception as e:
        logger.error(f"Error in show_model_auc_comparisons: {e}")
        raise

#----------------------------------------------------------------------------------------
def get_optimal_thr(clf:XGBClassifier, df:pd.DataFrame, tag:str, th_list_log:ArrayLike=np.linspace(-5,-0.01,200))->Tuple[pd.DataFrame, float, List]:
    """
    A helper function to calculate optimal threshold for maximizing F2-Score
    and providing data for plotting all metrics
    Args:
        - clf: classifier model
        - df: dataframe contains data for test with target column called 'Class'
        - tag: name for tracking
        - th_list_log: list of threshold for test in log10 format
    Returns:
        dataframe for plotting, optimal-threshold, and list of metrics for few selected thresold
    """
    try:
        y_test= df['Class']
        X_test = df.drop(columns=['Class'])
        #th_list_log=np.linspace(-5,-0.01,200)
        th_list=np.power(10,th_list_log)
        metrics=[]
        for th in tqdm(th_list,'Thr'):
            metrics.append(get_model_scores(X_test, y_test, 'Test', clf, th))
        metrics_df=pd.DataFrame(metrics)
        idx_max = metrics_df['F2-Score'].idxmax()
        thx=metrics_df.loc[idx_max,'Threshold']
        metrics_df['Threshold_log10']=np.log10(metrics_df['Threshold'])
        metrics_df=metrics_df.drop(columns=['Model', 'Threshold'])
        metrics2=[]
        for th in [thx,0.1, 0.15, 0.5, 0.85]: # selected thresholds !
            metrics2.append(get_model_scores(X_test, y_test, tag, clf, th))
        print(f'Maximizing F2-Score: Optimal Threshold= {thx:.4f} (log10= {np.log10(thx):.3f})')
        return metrics_df, thx, metrics2
    except Exception as e:
        logger.error(f"Error in get_optimal_thr: {e}")
        raise

#----------------------------------------------------------------------------------------
def find_optimal_triage_thresholds(clf, df, th_list_log,
                                     c_fn=10.0,
                                     c_fp=5.0,
                                     c_agent=0.2,
                                     agent_accuracy=0.90)->Tuple[float, float, pd.DataFrame, ArrayLike, ArrayLike, ArrayLike]:
    """
    A helper function to calculate optimal dual threshold for triage stage in agentic workflow
    and providing data for plotting cost contour to visualize 2D optimization
    Args:
        - clf: classifier model
        - df: dataframe contains data for test with target column called 'Class'
        - th_list_log: list of threshold for test in log10 format
        - c_tn: Cost of False Negative
        - c_fp: Cost of False Positive
        - c_agent: Cost of not-auto-detect (passing to LLM-agent)
        - agent_accuracy: implied cost of LLM-agent decision
    Returns:
        optimal thresholds, result dataframe, and meshgrid data for contour plot
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        y_true = df['Class'].values
        X_test = df.drop(columns=['Class'])
        y_probs = clf.predict_proba(X_test)[:, 1]

        thresholds = np.power(10, th_list_log)
        n_th = len(thresholds)

        # Tau_Low along X-axis (columns), Tau_High along Y-axis (rows)
        # In NumPy meshgrids generated by `np.meshgrid(x, y)`:
        #* Row index `i` maps to the **Y-axis** (`Tau_High`).
        #* Column index `j` maps to the **X-axis** (`Tau_Low`).
        Tau_Low_Log, Tau_High_Log = np.meshgrid(th_list_log, th_list_log)
        cost_surface = np.full((n_th, n_th), np.nan)

        best_cost = float('inf')
        best_tau_low, best_tau_high = None, None
        results = []

        # i = index for Tau_Low (X-axis), j = index for Tau_High (Y-axis)
        for i in tqdm(range(n_th), desc='Searching Log Thresholds'):
            tau_low = thresholds[i]

            for j in range(i + 1, n_th):
                tau_high = thresholds[j]

                # Categorize predictions
                auto_approve = y_probs < tau_low
                auto_block = y_probs > tau_high
                escalated = (y_probs >= tau_low) & (y_probs <= tau_high)

                # Costs
                fn_auto = np.sum((y_true == 1) & auto_approve) * c_fn
                fp_auto = np.sum((y_true == 0) & auto_block) * c_fp

                n_escalated = np.sum(escalated)
                agent_fn = np.sum((y_true == 1) & escalated) * (1 - agent_accuracy) * c_fn
                agent_fp = np.sum((y_true == 0) & escalated) * (1 - agent_accuracy) * c_fp

                total_cost = fn_auto + fp_auto + (n_escalated * c_agent) + agent_fn + agent_fp

                # [NOTE]: Matrix row index is j (Y-axis), column index is i (X-axis)
                cost_surface[j, i] = total_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_tau_low = tau_low
                    best_tau_high = tau_high

                results.append({
                    'tau_low': tau_low,
                    'tau_high': tau_high,
                    'log_tau_low': th_list_log[i],
                    'log_tau_high': th_list_log[j],
                    'automation_rate': (np.sum(auto_approve) + np.sum(auto_block)) / len(y_true),
                    'agent_load': n_escalated / len(y_true),
                    'total_cost': total_cost
                })

        return best_tau_low, best_tau_high, pd.DataFrame(results), Tau_Low_Log, Tau_High_Log, cost_surface
    except Exception as e:
        logger.error(f'Error in find_optimal_triage_thresholds: {e}')
        raise

def plot_optimal_triage_thresholds_contour(best_tau_low: float, best_tau_high:float, Tau_Low_Log:ArrayLike, Tau_High_Log:ArrayLike, cost_surface:ArrayLike):
    """
    plot 2D visualization of cost optimization in dual threshold using Contour plot
    Args:
        - best_tau_low: optimal threshold low
        - best_tau_high: optimal threshold high
        - Tau_Low_Log: X for Contour plot
        - Tau_High_Log: Y for Contour plot
        - cost_surface: Z for Contour plot
    Returns:
        None
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        # Plot setup
        fig, ax = plt.subplots(figsize=(9, 7))

        # Logarithmic contour levels prevent high-cost spikes from starving the rest of the plot
        min_val = np.nanmin(cost_surface)
        max_val = np.nanmax(cost_surface)
        levels = np.logspace(np.log10(min_val), np.log10(max_val), 20)

        # Filled contour background
        #logarithmic contour levels or LogNorm()
        cf = ax.contourf(Tau_Low_Log, Tau_High_Log, cost_surface, levels=levels, norm=LogNorm(), cmap="viridis_r")
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label("Total Cost (Log Scale)", rotation=270, labelpad=15)

        # Contour line overlays with labels
        CS = ax.contour(Tau_Low_Log, Tau_High_Log, cost_surface, levels=levels, norm=LogNorm(), colors="black", alpha=0.7, linewidths=0.5)
        ax.clabel(CS, fontsize=8, fmt="%1.0f")

        # Plot optimal point using log coordinates matching the plot axes
        opt_log_low = np.log10(best_tau_low)
        opt_log_high = np.log10(best_tau_high)


        # add value and log-value of optimal thresholds low and high on the plot
        ax.plot(opt_log_low, opt_log_high, 'r*', markersize=15,
                label=f"Optimal: τ_low={best_tau_low:.3f} (log10: {opt_log_low:.3f}")

        ax.plot(opt_log_low, opt_log_high, 'r*', markersize=15,
                label=f"Optimal: τ_high={best_tau_high:.3f} (log10: {opt_log_high:.3f}")


        # Diagonal boundary line (tau_low == tau_high in log-space)
        ax.plot([Tau_Low_Log.min(), 0], [Tau_High_Log.min(), 0], 'k--', alpha=0.5, label="Boundary (τ_low = τ_high)")

        ax.set_title("Triage Threshold Log-Space Cost Surface", fontsize=13, fontweight="bold")
        ax.set_xlabel("log10(τ_low) → Auto-Approve Threshold", fontsize=11)
        ax.set_ylabel("log10(τ_high) → Auto-Block Threshold", fontsize=11)
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
    except Exception as e:
        logger.error(f'Error in plot_optimal_triage_thresholds_contour: {e}')
        raise