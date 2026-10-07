"""
This file contains the utility functions to generate samples by
- Oversampling: Using SMOTE (Synthetic Minority Over-sampling Technique)
- Generative AI (GenAI) model: Using TVAE (Tabular Variational Auto-Encoder) from Synthetic Data Vault (SDV) library
- Generative AI (GenAI) model: Using CTGAN (Conditional Tabular Generative Adversarial Network) from (SDV) library
"""

from pathlib import Path
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from collections import Counter
from numpy.typing import ArrayLike # for type-hint
import pandas as pd
from typing import Tuple, Literal, List, Optional
from config import settings
from sklearn.preprocessing import RobustScaler
from sdv.metadata import Metadata, SingleTableMetadata
from sdv.single_table import TVAESynthesizer, CTGANSynthesizer
from sdv.utils import load_synthesizer
from sdmetrics.reports import QualityReport
import torch
import joblib
import numpy as np
from matplotlib.axes import Axes # for type-
import seaborn as sns

import logging
logger = logging.getLogger(__name__)

#----------------------------------------------------------------------------------------
def get_resample_bsmto(X_train:pd.DataFrame, y_train: pd.Series)->Tuple[pd.DataFrame, pd.Series]:
    """
    get resample (over-sample) data using BorderlineSMOTE
    Args:
        X_train: features of the train dataset in dataframe format
        y_train: target of the train dataset in series format
    Return:
        resampled data (balanced data set with equal number of samples in both classes)
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        # 1. Initialize BorderlineSMOTE
        # 'kind' can be 'borderline-1' (default) or 'borderline-2'
        bsmote = BorderlineSMOTE(kind='borderline-1', random_state=settings.RANDOM_SEED)

        # 2. Resample ONLY the training set
        X_train_res, y_train_res = bsmote.fit_resample(X_train, y_train)

        # 3. Verify class distribution before and after
        # print("Original train class distribution:", Counter(y_train))
        # print("Resampled train class distribution:", Counter(y_train_res))
        print(f'Class distribution, Original vs Resampled: {Counter(y_train)} ---> {Counter(y_train_res)}')

        return X_train_res, y_train_res
    except Exception as e:
        logger.error(f"Error in get_resample_bsmto: {e}")
        raise

#----------------------------------------------------------------------------------------
def get_resample_smto(X_train:pd.DataFrame, y_train: pd.Series)->Tuple[pd.DataFrame, pd.Series]:
    """
    get resample (over-sample) data using SMOTE
    Args:
        X_train: features of the train dataset in dataframe format
        y_train: target of the train dataset in series format
    Return:
        resampled data (balanced data set with equal number of samples in both classes)
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        # 1. Initialize standard SMOTE
        smote = SMOTE(sampling_strategy='auto', random_state=settings.RANDOM_SEED)

        # 2. Resample ONLY the training set
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # 3. Verify class distribution before and after
        # print("Original train class distribution:", Counter(y_train))
        # print("Resampled train class distribution:", Counter(y_train_res))
        print(f'Class distribution, Original vs Resampled: {Counter(y_train)} ---> {Counter(y_train_res)}')

        return X_train_res, y_train_res
    except Exception as e:
        logger.error(f"Error in get_resample_smto: {e}")
        raise

#----------------------------------------------------------------------------------------
def get_scaled_data_for_synthesizer(train_df:pd.DataFrame, class_label:Literal[0,1], synthesizer:Literal['tvae','ctgan'], ext:str='')->pd.DataFrame:
    """
    get scaled data, also save the scaler
    Args:
        train_df: dataframe of training data
        class_label: class label: 0=majority(legit) and 1=minority(fraud)
        synthesizer: tvae (TVAE) or ctgan (CTGAN) used for file name and tracking
        ext: extra extension for file name tracking
    Return:
        scaled dataframe used as the input to SDV synthesizer
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        label = 'minority' if class_label==1 else 'majority'
        scaler_path=settings.MODEL_DIR/f'{synthesizer}_{label}_scaler{ext}.pkl'

        target_col = 'Class'
        T=settings.HOURS_IN_TIME_GROUP
        categorical_cols=[f'Hour/{T}']

        class_df = train_df[train_df[target_col] == class_label].copy()
        feature_cols = [c for c in train_df.columns if c != target_col]
        continuous_cols = [c for c in feature_cols if c not in categorical_cols]

        scaler = RobustScaler()
        scaled_df = class_df.copy()
        scaled_df[continuous_cols] = scaler.fit_transform(class_df[continuous_cols])
        sdv_input = scaled_df[feature_cols].copy()
        joblib.dump(scaler, scaler_path)
        print(f'{synthesizer.upper()} Scaler for {label}-Class saved at {scaler_path}')
        return sdv_input
    except Exception as e:
        logger.error(f"Error in get_scaled_data_for_synthesizer: {e}")
        raise

#----------------------------------------------------------------------------------------
def get_metadata_for_synthesizer(sdv_input:pd.DataFrame, class_label:Literal[0,1], synthesizer:Literal['tvae','ctgan'])->Metadata:
    """
    prepare metadata , save it and return it for SDV synthesizer
    Args:
        sdv_input: dataframe used as the input to synthesizer
        class_label: class label: 0=majority(legit) and 1=minority(fraud)
        synthesizer: tvae (TVAE) or ctgan (CTGAN) used for file name and tracking
    Return:
        loaded metadata from saved json file
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        label = 'minority' if class_label==1 else 'majority'
        metadata_path = settings.METADATA_DIR/f'metadata_{synthesizer}_{label}.json'
        metadata = Metadata.detect_from_dataframe(data=sdv_input, table_name=f'{synthesizer}_{label}')
        T=settings.HOURS_IN_TIME_GROUP
        categorical_cols=[f'Hour/{T}']

        for col in categorical_cols:
            metadata.update_column(column_name=col, sdtype='categorical')

        if metadata_path.exists():
            metadata_path.unlink()
        metadata.save_to_json(filepath=metadata_path)
        metadata_loaded = Metadata.load_from_json(metadata_path)
        return metadata_loaded
    except Exception as e:
        logger.error(f"Error in get_metadata_for_synthesizer: {e}")
        raise

#----------------------------------------------------------------------------------------
def get_model_for_synthesizer(
        train_df:pd.DataFrame,
        class_label:Literal[0,1],
        synthesizer:Literal['tvae','ctgan'],
        ext:str='',
        n_epochs:Optional[int]=None,
        prn:bool=True
    )->TVAESynthesizer | CTGANSynthesizer:
    """
    load existing model, or train new model and return it
    Args:
        train_df: dataframe of training data
        class_label: class label: 0=majority(legit) and 1=minority(fraud)
        synthesizer: tvae (TVAE) or ctgan (CTGAN) used for file name and tracking
        ext: extra extension for file name
        n_epochs: number of epochs (if None we use default setting per case)
        prn: boolean to control some printing
    Return:
        SDV synthesizer model of type TVAE or CTGAN
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        if class_label not in [0, 1]:
            raise ValueError(f"Got class_label={class_label}. Valid=[0, 1]")

        if synthesizer not in ['tvae', 'ctgan']:
            raise ValueError(f"Got synthesizer={synthesizer}. Valid=['tvae', 'ctgan']")

        label = 'minority' if class_label==1 else 'majority'
        model_path=settings.MODEL_DIR/f'{synthesizer}_{label}_model{ext}.pkl'
        if model_path.exists():
            if prn: print(f'Loading {synthesizer.upper()} model for {label}-Class from {model_path}')
            model = load_synthesizer(model_path)
            return model
        else:
            if prn: print(f'Start building new {synthesizer.upper()} model for {label}-Class')

        sdv_input = get_scaled_data_for_synthesizer(train_df, class_label, synthesizer, ext)
        metadata_loaded = get_metadata_for_synthesizer(sdv_input, class_label, synthesizer)
        if synthesizer == 'tvae':
            if class_label == 1: # Minority Class
                model = TVAESynthesizer(
                    metadata=metadata_loaded,
                    #compress_dims=(128,64), #(64, 32), # default =(128,128)
                    #decompress_dims=(64, 128), #(32, 64), # default =(128,128)
                    #embedding_dim=64, #32, # default = 128
                    #l2scale=1e-3, # default = 1e-5
                    epochs=n_epochs or 100, # if None then 100 otherwise use n_epochs
                    batch_size=32,
                    enable_gpu=torch.cuda.is_available(),
                    verbose=True
                )
            else: # Majority Class
                model = TVAESynthesizer(
                    metadata=metadata_loaded,
                    compress_dims=(256,256), #(64, 32), # default =(128,128)
                    decompress_dims=(256, 256), #(32, 64), # default =(128,128)
                    #embedding_dim=64, #32, # default = 128
                    #l2scale=1e-3, # default = 1e-5
                    epochs=n_epochs or 200, # if None then 200 otherwise use n_epochs
                    batch_size=512,
                    enable_gpu=torch.cuda.is_available(),
                    verbose=True
                )
        else:
            if class_label == 1: # Minority Class
                model = CTGANSynthesizer(
                    metadata=metadata_loaded,
                    #generator_dim=(128, 128),
                    #discriminator_dim=(128, 128),
                    #embedding_dim=64,
                    #generator_decay=1e-4,
                    #discriminator_decay=1e-4,
                    pac=10,
                    epochs=n_epochs or 100, # if None then 100 otherwise use n_epochs
                    batch_size=30,
                    enable_gpu=torch.cuda.is_available(),
                    verbose=True
                )
            else: # Majority Class
                model = CTGANSynthesizer(
                    metadata=metadata_loaded,
                    #generator_dim=(128, 128), default 256x256
                    #discriminator_dim=(128, 128), default 256x256
                    #embedding_dim=64, default 128
                    #generator_decay=1e-4,
                    #discriminator_decay=1e-4,
                    pac=10,
                    epochs=n_epochs or 200, # if None then 200 otherwise use n_epochs
                    batch_size=500, # should be divisible by pac=10
                    enable_gpu=torch.cuda.is_available(),
                    verbose=True
                )

        model.fit(sdv_input)
        model.save(model_path)
        if prn: print(f'{synthesizer.upper()} Model for {label}-Class saved at {model_path}')
        return model
    except Exception as e:
        logger.error(f"Error in get_model_for_synthesizer: {e}")
        raise

#----------------------------------------------------------------------------------------
def get_sample_form_synthesizer(
        model:TVAESynthesizer | CTGANSynthesizer,
        num_rows:int,
        feature_cols:List[str],
        class_label:Literal[0,1],
        synthesizer:Literal['tvae','ctgan'],
        ext:str='',
        prn: bool = True
    )->pd.DataFrame:
    """
    get scaled data, and save the scaler
    Args:
        model: SDV synthesizer model of type TVAE or CTGAN
        num_rows: number of rows/samples to generate
        feature_cols: feature colums name in original data
        class_label: class label: 0=majority(legit) and 1=minority(fraud)
        synthesizer: tvae (TVAE) or ctgan (CTGAN) used for file name and tracking
        ext: extra extension for file name tracking
        prn: boolean to control print
    Return:
        dataframe of generated samples
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        label = 'minority' if class_label==1 else 'majority'
        scaler_path=settings.MODEL_DIR/f'{synthesizer}_{label}_scaler{ext}.pkl'
        if prn: print(f'Loading {synthesizer.upper()} Scaler for {label}-Class from {scaler_path}')
        scaler = joblib.load(scaler_path)

        target_col = 'Class'
        T=settings.HOURS_IN_TIME_GROUP
        categorical_cols=[f'Hour/{T}']

        if prn: print(f'Start Generating {num_rows} samples ...')

        synth_scaled = model.sample(num_rows=num_rows)
        synth_rescaled = synth_scaled.copy()

        continuous_cols = [c for c in feature_cols if c not in categorical_cols]
        synth_rescaled[continuous_cols] = scaler.inverse_transform(synth_scaled[continuous_cols])

        for col in categorical_cols:
            synth_rescaled[col] = synth_rescaled[col].astype(int)

        synth_rescaled[target_col] = class_label
        return synth_rescaled
    except Exception as e:
        logger.error(f"Error in get_sample_form_synthesizer: {e}")
        raise

#----------------------------------------------------------------------------------------
def plot_dist_comparison(
        df:pd.DataFrame, # ORG Data
        dft:pd.DataFrame, # Gen-Data By TVAE
        dfc:pd.DataFrame, # Gen-Data By CTGAN
        col_name:str,
        class_label:Literal[0,1],
        ax:Axes,
        plot_type:Literal['box','violin']='box'
    ):
    """
    compare distribution among original-data, and generated data by TVAE and CTGAN
    Args:
        df: dataframe of original data
        dft: dataframe of generated data by TVAE
        dfc: dataframe of generated data by CTGAN
        col_name: name of the column/feature to plot its distribution
        class_label: class label: 0=majority(legit) and 1=minority(fraud)
        ax: provided axes for plotting
        plot_type: box-plot or violin-plot
    Return:
        None
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        title_ex = 'Fraud-class' if class_label==1 else 'Legit-class'
        org = df[df['Class'] == class_label][[col_name]].copy()
        org['Group']='ORG'

        tvae = dft[dft['Class'] == class_label][[col_name]].copy()
        tvae['Group']='TVAE'

        ctgan = dfc[dfc['Class'] == class_label][[col_name]].copy()
        ctgan['Group']='CTGAN'

        combined_df = pd.concat([org, tvae, ctgan], axis=0)

        palette = {'ORG': 'tab:green', 'TVAE':'tab:blue', 'CTGAN':'tab:red'}
        order=['TVAE', 'ORG', 'CTGAN'] # ORG in the middle

        if plot_type == 'violin':
            sns.violinplot(
                data=combined_df, x='Group', y=col_name, ax=ax,
                palette=palette, inner='quartile', cut=0, hue='Group', legend=False, order=order
            )
        elif plot_type == 'box':
            sns.boxplot(
                data=combined_df, x='Group', y=col_name, ax=ax,
                palette=palette, showfliers=False, hue='Group', legend=False, order=order
            )
        else:
            raise ValueError("plot_type must be either 'violin' or 'box'")

        #ax.set_title(f"Distribution Comparison: {col_name}", fontsize=11, fontweight='bold')
        ax.set_title(f"{col_name}-{title_ex} Dist.") #, fontsize=11, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("")# ("Scaled Value")
        ax.grid(True, linestyle='--', alpha=0.3)
    except Exception as e:
        logger.error(f"Error in plot_dist_comparison: {e}")
        raise

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#----------------------------------------------------------------------------------------
def get_metadata_dic_for_quality_report(data: pd.DataFrame)->dict:
    """
    Get metadata with proper structure for sdmetrics library, to to be used in quality report
    Args:
        - data: a dataframe without Class columns
    Returns:
        The metadata dictionary
    Raise:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        metadata_obj = SingleTableMetadata()
        metadata_obj.detect_from_dataframe(data=data)

        # Explicitly set sdtypes for special columns
        T=settings.HOURS_IN_TIME_GROUP
        metadata_obj.update_column(column_name=f'Hour/{T}', sdtype='categorical')
        #metadata_obj.update_column(column_name='Class', sdtype='boolean') # it's constant not needed

        # Wrap into the multi-table metadata contract required by QualityReport
        metadata_dict = {
            'tables': {
                'transactions': metadata_obj.to_dict()
            }
        }
        return metadata_dict
    except Exception as e:
        logger.error(f'Error in get_metadata_dic_for_quality_report: {e}')
        raise

#----------------------------------------------------------------------------------------
def evaluate_model_quality(
        real_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        metadata: dict,
        model_name: str,
        class_label:Literal[0,1])->Tuple[float, pd.DataFrame]:
    """
    Generates an SDMetrics QualityReport and extracts scores.
    Args:
        - real_data: dataframe of original/real data without Class columns
        - synth_data: dataframe of generated data without Class columns
        - metadata: dict of metadata
        - model_name: model name for tracking purpose
        - class_label: 0: Legit, 1: Fraud
    Returns:
        The metadata dictionary
    Raise:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        label_ext = '(Fraud Class)' if class_label == 1 else '(Legit Class)'
        report = QualityReport()

        # Wrap DataFrames into table dictionaries expected by SDMetrics
        real_tables = {'transactions': real_data}
        synth_tables = {'transactions': synth_data}

        # Generate report silently or with progress
        report.generate(real_tables, synth_tables, metadata, verbose=False)

        # Extract overall score & details DataFrame
        overall_score = report.get_score()
        # Extract details DataFrames (using capitalized SDMetrics column names)
        # Table 	Column 	Metric 	Score
        # Metrics= KSComplement for cont.-var like V1-V28, and TVComplement for ordinal like Hour/2
        shapes_df = report.get_details(property_name='Column Shapes', table_name='transactions')
        shapes_df=shapes_df.drop(columns=['Table','Metric']).copy()

        shapes_df=shapes_df.rename(columns={'Score': f'{model_name}_Score'})

        #Table 	Column 1 	Column 2 	Metric 	Score 	Real Correlation 	Synthetic Correlation 	Real Association 	Meets Threshold?
        #n*(n+1) rows = 30*31/2=435 (we do not use this, many nan,...)
        #trends_df = report.get_details(property_name='Column Pair Trends', table_name='transactions')

        #Cardinality is not defined for single table like our case


        print(f"=== {model_name} {label_ext} Quality Report ===")
        print(f"Overall Quality Score: {overall_score * 100:.2f}%\n")
        return  overall_score, shapes_df

    except Exception as e:
        logger.error(f'Error in evaluate_model_quality: {e}')
        raise

