"""
This file contains the utility functions used to get test case samples for testing in this project
"""

from config import settings
from typing import Literal
import pandas as pd
import logging
logger = logging.getLogger(__name__)

label_map = {'tp': 'TRUE_POSITIVE', 'tn': 'TRUE_NEGATIVE', 'fn':'FALSE_NEGATIVE','fp': 'FALSE_POSITIVE'}
node_map = {'det':'Detector', 'llm': 'LLM_Agent'}

def get_test_case_sample(
        source_id: Literal['real', 'gen'],
        node_id: Literal['det','llm'],
        label_id: Literal['tn','fp','fn', 'tp'],
        test_id: int
    )->tuple[pd.DataFrame | None, int]:
    """
    """
    try:
        file_path = settings.DATA_DIR/f'labeled_dataset_from_{source_id}_data_test.csv'
        df=pd.read_csv(file_path)
        df_filtered=df[(df['label']==label_map[label_id]) & (df['node']==node_map[node_id])].reset_index(drop=True)
        df_filtered = df_filtered.drop(columns=['fraud_prob', 'label', 'node'])
        max_index = len(df_filtered)
        if 0 < test_id <= max_index:
            return df_filtered.iloc[[test_id-1]], max_index # return as dataframe
        elif test_id == -1:
            return df_filtered, max_index
        else:
            return None, max_index
    except Exception as e:
        logger.error(f'Error in get_test_case_sample: {e}')
        raise


