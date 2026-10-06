import shutil
from pathlib import Path
import chromadb
from config import settings
#from numpy.typing import ArrayLike # for type-hint
import pandas as pd
import numpy as np
import logging
import hashlib
from typing import Any, Dict, List, Union, Tuple
logger = logging.getLogger(__name__)

#----------------------------------------------------------------------------------------
def get_or_create_vector_db_collection(
    persist_directory: Path = settings.CHROMA_DIR,
    collection_name: str = settings.COLLECTION_NAME,
) -> chromadb.Collection:
    """Returns an existing Chroma collection instance if it exists, otherwise initializes a new one.
    Args:
        persist_directory (Path): Main directory path for Chroma DB storage.
        collection_name (str): Name of the Chroma collection to retrieve or create.
    Returns:
        chromadb.Collection: Chroma database collection instance.
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        client = chromadb.PersistentClient(path=str(persist_directory))

        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return collection

    except Exception as e:
        logger.error(f"Error in get_or_create_vector_db_collection: {e}")
        raise

#----------------------------------------------------------------------------------------
def reset_vector_db_on_disk(persist_directory: Path = settings.CHROMA_DIR):
    """Deletes the entire Chroma persistence directory from disk.
    Args:
        persist_directory (Path): Main directory path for Chroma DB storage.
    Returns:
        None
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        path = Path(persist_directory)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            print(f"Successfully deleted vector database at {path}")
        else:
            print(f"No database found at {path}")
    except Exception as e:
        logger.error(f"Error in reset_vector_db_on_disk: {e}")
        raise

#----------------------------------------------------------------------------------------
def get_id_and_embedding_vector(features: pd.DataFrame) -> tuple[list[list[float]], list[str]]:
    """
    Converts dataframe rows into embedding vectors and corresponding deterministic IDs.
    Processes V1-V28, Z-score standardizes Amount, and converts Hour-block into
    sine/cosine cyclical components.
    Args:
        features: Input dataframe contains all features column in cleaned dataset order
    Returns:
        list of embedding vectors, and list of ids for them
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        df_columns = pd.read_csv(settings.DATA_DIR / settings.DATASET_FILE_CLEANED, nrows=0).columns
        feature_cols = [c for c in df_columns if c != "Class"]
        # 1. Validate that we have  30 feature columns (V1 to V28, Amount, Hour/2)
        # Convert df.columns to a list and compare directly against feature_cols
        if list(features.columns) != feature_cols:
            raise ValueError(
                f"Input DataFrame columns do not match expected feature columns in exact order.\n"
                f"Expected ({len(feature_cols)}): {feature_cols}\n"
                f"Received ({len(features.columns)}): {list(features.columns)}"
            )
        features = features.astype(float).copy()
        T = settings.HOURS_IN_TIME_GROUP
        HourT=f'Hour/{T}'
        features['Hour_sin']=features[HourT]

        # 2. Extract and standardize Amount (col index 28)
        amount_scaled = (features.iloc[:, 28] - settings.AMOUNT_MEAN) / settings.AMOUNT_STD
        features.iloc[:, 28] = amount_scaled

        # 3. Compute Cyclical Time Features (col index 29)
        max_val = (23 // T) + 1
        scale = 2 * np.pi / max_val

        #features.iloc[:, 29] = features.iloc[:, 29].astype(np.float64)

        #features[HourT] = features[HourT].astype(float)
        #features.loc[:, HourT] = features[HourT].astype(float)
        hour_vals = features.iloc[:, 29].values
        cos_hour = np.cos(scale * hour_vals)
        sin_hour = np.sin(scale * hour_vals)

        # Replace col 29 with cos_hour and append sin_hour as 31st feature
        features.iloc[:, 29] = cos_hour
        #features["Hour_sin"] = sin_hour
        #features.loc[:, "Hour_sin"] = sin_hour
        features.iloc[:, 30] = sin_hour

        # 4. Round all feature values to 6 decimals
        embeddings_np = features.to_numpy().round(6)

        # 5. Generate deterministic IDs using SHA256 hashes of formatted strings
        ids = [
            hashlib.sha256("_".join(map(str, row)).encode("utf-8")).hexdigest()[-10:]
            for row in embeddings_np
        ]

        return embeddings_np.tolist(), ids

    except Exception as e:
        logger.exception("Error in get_id_and_embedding_vector: %s", e)
        raise

#----------------------------------------------------------------------------------------
def init_vector_db_with_existing_historical_data(tag: str, batch_size: int = 50) -> pd.DataFrame:
    """
    Loads historical labeled CSV dataset, extracts features and metadata,
    and ingests embeddings into ChromaDB collection.
    Args:
        tag: Refer to historical data file currently real_data and gen_data
        batch_size: batch size to be used for efficiency and also handling large files
    Returns:
        return loaded df for test purpose (can be ignored if not needed)
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        csv_name = f"labeled_dataset_from_{tag}.csv"
        csv_path = settings.DATA_DIR / csv_name
        print(f'Init using {tag} with labeled data at: {csv_path}')

        if csv_path.exists():
            logger.info(f"Loading existing labeled dataset from: {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            raise ValueError(f"No labeled dataset found for given tag: '{tag}' at path: {csv_path}")

        # 1. Filter out Detector True Negatives (kept as sample rows)
        detector_tn_mask = (df["label"] == "TRUE_NEGATIVE") & (df["node"] == "Detector")
        df = df[~detector_tn_mask].copy()

        if df.empty:
            logger.warning(f"No data remaining after filtering for tag '{tag}'. Skipping ingestion.")
            return

        # 2. Align feature columns for embedding generation
        df_columns = pd.read_csv(settings.DATA_DIR / settings.DATASET_FILE_CLEANED, nrows=0).columns
        feature_cols = [c for c in df_columns if c != "Class"]

        # Generate vector embeddings and SHA256 deterministic IDs
        embedding_vectors, ids = get_id_and_embedding_vector(df[feature_cols])

        # 3. Build metadata list from extra columns
        metadatas = []
        for _, row in df.iterrows():
            meta = {
                "label": str(row.get("label", "UNKNOWN")),
                "node": str(row.get("node", "UNKNOWN")),
                "actual_class": "Fraud" if row.get("Class", 0) == 1 else "Legit",
                "fraud_prob": float(row.get("fraud_prob", 0.0)),
                "Amount": float(row.get("Amount", 0.0)),
                "Hour_Block": int(row.get("Hour/2", 0)),
            }
            metadatas.append(meta)

        # 4. Get Chroma collection instance
        collection = get_or_create_vector_db_collection()

        # 5. Ingest into Chroma DB in batches using upsert
        total_records = len(ids)
        logger.info(f"Ingesting {total_records} records into collection '{collection.name}'...")

        for i in range(0, total_records, batch_size):
            batch_end = min(i + batch_size, total_records)
            collection.upsert(
                ids=ids[i:batch_end],
                embeddings=embedding_vectors[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )

        logger.info(f"Successfully upserted {total_records} items into vector DB.")
        print(f'Init using {tag} is DONE!')
        return df # use returning df for test
    except Exception as e:
        logger.error(f'Error in init_vector_db_with_existing_historical_data: {e}')
        raise


def query_top_n_closest_from_df(
    raw_df_row: pd.DataFrame,
    top_n: int = 5,
    where_filter: Union[Dict[str, Any], None] = None,
    collection: Union[chromadb.Collection, None] = None,
) -> tuple[pd.DataFrame,str]:
    """Transforms a raw transaction DataFrame row into its custom feature embedding,

    queries ChromaDB for top N nearest historical records, and returns a formatted report.

    Args:
        raw_df_row (pd.DataFrame): 1-row DataFrame containing raw transaction features
          (V1-V28, Amount, Hour/2).
        top_n (int): Number of nearest neighbors to return.
        where_filter (Dict, optional): Chroma metadata filter expression.
        collection (chromadb.Collection, optional): Existing collection instance.

    Returns:
        pd.DataFrame: Formatted report with IDs, distance, similarity score, and metadata.
    """
    try:
        # 1. Align feature columns (exclude 'Class' if present)
        df_columns = pd.read_csv(settings.DATA_DIR / settings.DATASET_FILE_CLEANED, nrows=0).columns
        feature_cols = [c for c in df_columns if c != "Class"]

        # Ensure input row contains required feature columns
        query_features_df = raw_df_row[feature_cols].copy()

        # 2. Transform raw features using the exact same embedding transformation function
        embedding_vectors, query_ids = get_id_and_embedding_vector(query_features_df)
        query_vector = embedding_vectors[0]  # Take the 31-element feature vector for row 0

        # 3. Retrieve or default collection
        if collection is None:
            collection = get_or_create_vector_db_collection()

        # 4. Query ChromaDB using the transformed feature vector
        query_kwargs = {
            "query_embeddings": [query_vector],
            "n_results": top_n,
            "include": ["metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = collection.query(**query_kwargs)

        # 5. Extract and format results
        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        report_data = []
        for doc_id, dist, meta in zip(ids, distances, metadatas):
            # record = {
            #     "neighbor_id": doc_id,
            #     "cosine_distance": round(dist, 6),
            #     #"similarity_score": round(1.0 - dist, 6),
            #     **meta,
            # }
            #neighbor_id 	cosine_distance 	similarity_score 	actual_class 	node 	Amount 	label 	fraud_prob 	Hour_Block
            record={
                'id': doc_id,
                'distance': round(dist, 6),
                'Amount': round(meta.get('Amount',0),2),
                'Hour_Block': int(meta.get('Hour_Block',0)),
                'fraud_prob': round(meta.get('fraud_prob',0.5),6),
                'actual_class': meta.get('actual_class','UNKNOWN'),
                'label': meta.get('label','UNKNOWN'),
                'node': meta.get('node','UNKNOWN'),
            }
            report_data.append(record)

        return pd.DataFrame(report_data), query_ids[0]

    except Exception as e:
        logger.error(f'Error in query_top_n_closest_from_df: {e}')
        raise



def init_ltm_db(force_reset:bool = False):
    """
    init and Create a new Chroma dB Vector dB as LTM database
    Args:
        - force_reset: if True, reset existing dB and then init/setup, otherwise no action is taken
    Returns:
        None
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        db_path = settings.CHROMA_DIR/settings.CHROMA_DB #'chroma.sqlite3'
        if db_path.exists():
            if force_reset:
                reset_vector_db_on_disk()
                print(f"Existing database at {db_path} removed.")
            else:
                print(f'database already exist at :{db_path} (no action is taken)')
                return
        else:
            print(f'No database at {db_path}, Init and setup for the first time!')

        _ = init_vector_db_with_existing_historical_data('real_data')
        _ = init_vector_db_with_existing_historical_data('gen_data')
    except Exception as e:
        logger.error(f"Error in init_ltm_db: {e}")
        raise


if __name__ == "__main__":
    init_ltm_db(force_reset=True)

