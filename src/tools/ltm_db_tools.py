# ltm_db_tools.py (Chroma dB Vector dB used for LTM)

from typing import Literal, Union
import pandas as pd
import chromadb
from config import settings
import logging
from db_model.ltm_model import (
    get_id_and_embedding_vector,
    get_or_create_vector_db_collection,
    query_top_n_closest_from_df,
)
from langchain_core.tools import tool

logger = logging.getLogger(__name__)



def add_single_transaction_to_vector_db(
    df: pd.DataFrame,
    fraud_prob: float,
    actual_class: Union[int, str],
    actual_label: str,
    node: Literal["Detector", "LLM_Agent","Human_Agent"],
    collection: Union[chromadb.Collection, None] = None,
) -> tuple[bool,str]:
    """Transforms a single transaction DataFrame row and adds/upserts it into ChromaDB.

    Args:
        df (pd.DataFrame): 1-row DataFrame containing raw features (V1-V28, Amount, Hour/2).
        fraud_prob (float): Model predicted fraud probability (e.g., 0.852).
        actual_class (int | str): Class indicator (0 or 'Legit', 1 or 'Fraud').
        actual_label (str): Performance label ('TRUE_POSITIVE', 'FALSE_POSITIVE', etc.).
        node (str): Decision origin node ('Detector' or 'LLM_Agent').
        collection (chromadb.Collection, optional): Chroma collection instance.

    Returns:
        bool: True if the item was newly inserted, False if an existing item was updated. and the doc_id
    """
    try:
        # Ensure input is a single-row DataFrame
        if len(df) != 1:
            raise ValueError(f"Expected a 1-row DataFrame, but got {len(df)} rows.")

        # 1. Align feature columns (exclude target column 'Class' if present)
        df_columns = pd.read_csv(settings.DATA_DIR / settings.DATASET_FILE_CLEANED, nrows=0).columns
        feature_cols = [c for c in df_columns if c != "Class"]

        single_row_df = df[feature_cols].copy()

        # 2. Generate embedding vector and deterministic 10-char SHA256 ID
        embedding_vectors, ids = get_id_and_embedding_vector(single_row_df)
        doc_id = ids[0]
        embedding_vector = embedding_vectors[0]

        # 3. Format metadata dictionary
        # Convert class indicator to standard string format ('Fraud' or 'Legit')
        if isinstance(actual_class, (int, float)):
            class_str = "Fraud" if int(actual_class) == 1 else "Legit"
        else:
            class_str = str(actual_class)

        raw_row = single_row_df.iloc[0]
        metadata = {
            "label": str(actual_label),
            "node": str(node),
            "actual_class": class_str,
            "fraud_prob": float(fraud_prob),
            "Amount": float(raw_row["Amount"]),
            "Hour_Block": int(raw_row["Hour/2"]),
        }

        # 4. Get default collection if not provided
        if collection is None:
            collection = get_or_create_vector_db_collection()

        # 5. Check if the document ID already exists prior to upsert
        existing = collection.get(ids=[doc_id], include=[])
        is_new_insert = len(existing["ids"]) == 0

        # 6. Upsert the item into ChromaDB
        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding_vector],
            metadatas=[metadata],
        )

        status_str = "Inserted new" if is_new_insert else "Updated existing"
        logger.info(f"{status_str} transaction record in vector DB (ID: {doc_id}).")

        return is_new_insert, doc_id

    except Exception as e:
        logger.error(f'Error in add_single_transaction_to_vector_db: {e}')
        raise

@tool
def get_top_n_context(transaction_data:dict['str', float], top_n:int = 3)->str:
    """
    get top_n closest-cosine-distance neighbors from vector-database and format as proper context for LLM-agent
    distance is between [0,2] with 0 is the best (exact match), and 2 is the worst
    Args:
        - transaction_data: transaction data in dict format [V1 tp V28 and Amount, Hour/2]
        - top_n: number of top closest results to be reported
    Return:
        report in string format
    """
    try:
        resp_str, _= get_top_n_context_for_human(transaction_data, top_n)
        return resp_str
    except Exception as e:
        logger.error(f'Error in get_top_n_context: {e}')
        return "No document found"


# avoid mixing tool-usage and function call
def get_top_n_context_for_human(transaction_data:dict['str', float], top_n:int = 3)->tuple[str, pd.DataFrame]:
    """
    get top_n closest-cosine-distance neighbors from vector-database and format as proper context for LLM-agent
    distance is between [0,2] with 0 is the best (exact match), and 2 is the worst
    Args:
        - transaction_data: transaction data in dict format [V1 tp V28 and Amount, Hour/2]
        - top_n: number of top closest results to be reported
    Return:
        report in string format, and dataframe format
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        single_row_df = pd.DataFrame([transaction_data])
        resp_df, qid = query_top_n_closest_from_df(single_row_df, top_n=top_n)
        #id 	distance 	Amount 	Hour_Block 	fraud_prob 	actual_class 	label 	node
        #label and node is for human, not agent, we may include label if agent suppose to do other analysis (not in this project)
        resp_df =resp_df.drop(columns=['label','node'])
        resp_str=''
        for j, r in resp_df.iterrows():
            resp_str +=f'[doc-{j+1}]-' + ", ".join(f"{k}: {v}" for k, v in r.items())+"\n"
        return resp_str, resp_df
    except Exception as e:
        logger.error(f'Error in get_top_n_context_for_human: {e}')
        raise

