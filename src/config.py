"""Project config file"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from pydantic import computed_field
import numpy as np
import pandas as pd
import logging
import json
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
import argparse
import joblib

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT_PATH / ".env" # very critical, otherwise work in jupyter but not in script

class Settings(BaseSettings):
    # --- 1. DIRECT FIELDS ----------------------------------------------------
    PROJECT_ROOT: Path = PROJECT_ROOT_PATH
    #PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    DATASET_ID: str = "mlg-ulb/creditcardfraud"
    DATASET_DIR:str = ""
    DATASET_FILE: str = "creditcard.csv"
    DATASET_FILE_CLEANED:str = "cleaned_creditcard.csv"
    DATASET_FILE_CLEANED_TRAINING:str = "cleaned_creditcard_training.csv"
    DATASET_FILE_CLEANED_TEST:str = "cleaned_creditcard_test.csv"
    DATASET_FILE_CLEANED_SAMPLE:str = "cleaned_dataset_sample.csv"


    RANDOM_SEED:int = 6488
    RNG: np.random.Generator = np.random.default_rng(RANDOM_SEED)

    MODEL_NAME: str   #=  'MODEL_NAME' # will load MODEL_NAME from .env
    OPENAI_API_KEY: str #= 'OPENAI_API_KEY' # will load OPENAI_API_KEY from .env
    OPENAI_BASE_URL: str # # will load OPENAI_API_KEY from .env
    TEMPERATURE: float # = 0.0 # will load TEMPERATURE from .env and convert to float
    USE_CONTEXT:bool #will load USE_CONTEXT from .env and convert to boolean
    #
    TRANSACTION_DB:str = 'transaction.db'
    LTM_DB:str = 'chroma.sqlite3' # Chroma Vector dB default name
    CHROMA_DB:str = 'chroma.sqlite3' # Chroma Vector dB default name (alias for LTM)
    CHECKPOINT_DB:str = 'checkpoint_db'
    WORKFLOW_GRAPH: str = 'workflow_graph.png'
    COLLECTION_NAME:str = "fraud_30d_direct" # Collection name in Chroma Vector dB
    #
    HOURS_IN_TIME_GROUP:int = 2 # after converting time to Hours in [0,23], we group them together [0,1]->0, [2,3]-> 1,...

    # when pkl file exist these value will be updated by values in pkl file
    LOW_THR: float = 0.026759
    HIGH_THR: float = 0.870154
    AMOUNT_MEAN: float = 89.3
    AMOUNT_STD: float = 251.4


    # --- 2. COMPUTED FIELDS --------------------------------------------------
    @computed_field
    @property
    def DATA_DIR(self) -> Path:
        return self.PROJECT_ROOT / "data"

    @computed_field
    @property
    def OUT_DIR(self) -> Path:
        return self.PROJECT_ROOT / "out"

    @computed_field
    @property
    def IMG_DIR(self) -> Path:
        return self.PROJECT_ROOT / "report/latex/img"

    @computed_field
    @property
    def DATASET_PATH(self) -> Path:
        if self.DATASET_DIR:
            return  self.DATA_DIR/self.DATASET_DIR/self.DATASET_FILE
        else:
            return  self.DATA_DIR/self.DATASET_FILE


    @computed_field
    @property
    def LOG_DIR(self) -> Path:
        return self.OUT_DIR / "logs"

    @computed_field
    @property
    def CHECKPOINT_DIR(self) -> Path:
        return self.OUT_DIR / "db_checkpoint"

    @computed_field
    @property
    def LTM_DIR(self) -> Path:
        return self.OUT_DIR / "db_ltm"

    @property
    def CHROMA_DIR(self) -> Path:
        """Alias for LTM_DIR."""
        return self.LTM_DIR

    @computed_field
    @property
    def TRANSACTION_DIR(self) -> Path:
        return self.OUT_DIR / "db_transaction"

    @computed_field
    @property
    def MODEL_DIR(self) -> Path:
        return self.DATA_DIR / "ml_models" #we save all models in data if not needed for git add to gitignore

    @computed_field
    @property
    def METADATA_DIR(self) -> Path:
        return self.OUT_DIR / "sdv_metadata" # temporary metadata used by SDV library for TVAE and CTGAN


    # --- 3. CONFIGURATION ----------------------------------------------------
    # Configuration for the settings class
    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        extra="ignore",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- 4. METHODS ----------------------------------------------------------
    def create_directories(self):
        for path in [self.DATA_DIR,
        self.OUT_DIR,
        self.IMG_DIR,
        self.LOG_DIR,
        self.CHECKPOINT_DIR,
        self.LTM_DIR,
        self.TRANSACTION_DIR,
        self.MODEL_DIR,
        self.METADATA_DIR
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def check_all_db_exist(self):
        db_paths = {
            "LTM DB": self.LTM_DIR / self.LTM_DB,
            "Transaction DB": self.TRANSACTION_DIR / self.TRANSACTION_DB,
            "Checkpoint DB": self.CHECKPOINT_DIR / self.CHECKPOINT_DB
        }
        for db_name, db_path in db_paths.items():
            if not db_path.exists():
                print(f"WARNING : Missing database: {db_name} at {db_path}")


    def update_settings_from_pkl_file(self):
        amount_stat_path = self.DATA_DIR/'amount_mean_std.pkl'
        if amount_stat_path.exists():
            #print('Load and update amount-stat')
            stat = joblib.load(amount_stat_path)
            self.AMOUNT_MEAN = stat.get('mean', 89.3)
            self.AMOUNT_STD = stat.get('std', 251.4)
        dual_thr_path = self.DATA_DIR/'dual_thresholds.pkl'
        if dual_thr_path.exists():
            #print('load and update dual thr')
            thr = joblib.load(dual_thr_path)
            self.LOW_THR = thr.get('tau_low', 0.02676)
            self.HIGH_THR = thr.get('tau_high', 0.87015)

#------------------------------------------------------------------------------
# Create a single instance to be used everywhere
settings = Settings()
settings.create_directories()
#settings.check_all_db_exist()
settings.update_settings_from_pkl_file()


#------------------------------------------------------------------------------
class CustomLogger:
    def __init__(self, log_dir: Path = settings.LOG_DIR, save_format: str='jsonl', verbose: bool = False):
        self.log_dir = log_dir
        self.save_format = save_format
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.save_format =="jsonl":
            self.log_path = self.log_dir / "log.jsonl"
        else:
            self.log_path = self.log_dir / "log.txt"
        self._setup_logger(verbose)

    def _setup_logger(self, verbose: bool):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # Clear existing handlers to prevent duplicate logs if re-initialized
        if logger.hasHandlers():
            logger.handlers.clear()

        # 1. Rotating File Handler (Appends to existing file)
        # maxBytes=2MB, backupCount=5 (keeps 5 old files)
        file_handler = RotatingFileHandler(
            #self.log_path, maxBytes=2*1024*1024, backupCount=5
            self.log_path, maxBytes=100*1024, backupCount=3  # set it to 100K with 3 rotating
        )
        if self.save_format == "jsonl":
            file_handler.setFormatter(self._get_jsonl_formatter())
        else:
            # Use 0 or None for flat lines in the text file
            file_handler.setFormatter(self._get_txt_formatter(json_indent=None))
        logger.addHandler(file_handler)

        # 2. Human Readable Console (Optional)
        if verbose:
            console_handler = logging.StreamHandler(sys.stdout)
            # Use 2 for pretty-printing in the terminal
            console_handler.setFormatter(self._get_txt_formatter(json_indent=2))
            logger.addHandler(console_handler)

    def _get_jsonl_formatter(self):
        """Format the log record as a JSON string for the .jsonl file."""
        class JSONLFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "mod": record.name,
                    "file": record.filename,
                    "line": record.lineno,
                    #"message": record.getMessage() # handle separatly to cover both str and dict
                }
                # note on record.msg vs record.getMessage()
                # record.msg = obj = Any (String, Dict, Object, etc.)= placeholder
                # record.getMessage()=The result of processing that object.=str =placeholder already filled
                if isinstance(record.msg, dict):
                    log_data.update(record.msg) # to handle we we use another dict like part of AgentState
                else:
                    log_data["msg"] = record.getMessage()
                return json.dumps(log_data, default=str)

        return JSONLFormatter()


    def _get_txt_formatter(self, json_indent: int | None = None):
            """Returns a formatter that stringifies dicts with optional indentation."""
            class DynamicTextFormatter(logging.Formatter):
                def format(self, record):
                    orig_msg = record.msg
                    if isinstance(record.msg, dict):
                        # Use the indent passed to the factory (None/0 for files, 2 for console)
                        record.msg = json.dumps(record.msg, indent=json_indent, default=str)

                    result = super().format(record)
                    record.msg = orig_msg  # Keep the record clean for other handlers
                    return result

            # Define the string structure for text logs
            return DynamicTextFormatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s'
            )

#------------------------------------------------------------------------------
def get_epilog_msg():
    """
    get test-case-index info, to be append at the end of help in arg-parser
    """
    dfi=pd.read_csv(settings.DATA_DIR/'labeled_dataset_from_real_data_split_info.csv')
    dfi=dfi[dfi['node_split'].str.endswith('-Test')]
    dfi['node_split']=dfi['node_split'].str.replace('-Test','-real')
    dfi=dfi.rename(columns={'node_split':'node_source'})

    dfic=pd.read_csv(settings.DATA_DIR/'labeled_dataset_from_gen_data_split_info.csv')
    dfic=dfic[dfic['node_split'].str.endswith('-Test')]
    dfic['node_split']=dfic['node_split'].str.replace('-Test','-gen')
    dfic=dfic.rename(columns={'node_split':'node_source'})

    dfix=pd.concat([dfi, dfic])
    a=dfix.to_markdown(index=False)
    output=(
        f"{'+' * 95}\n"
        f'test (test-index) ranges depend on the source, node, and labels combinations as shown below:\n'
        f'e.g. 4 means: available indexes are 1,2,3,4, and 0 means no test-case available\n'
        f"{'+' * 95}\n"
        f'\n\n'
        f'{a}\n'
        f"{'+' * 95}\n"
    )
    return output

def get_args():
    """
    Handles command-line arguments.
    Easily expandable by adding more parser.add_argument() calls.
    """
    parser = argparse.ArgumentParser(
        description="Agentic Fraud Detection",
        epilog=get_epilog_msg(),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Verbose flag: store_true makes it a boolean (True if present, False if not)
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (pretty-print logs to console)"
    )

    # Example of an expandable argument for the log format
    parser.add_argument(
        "-f", "--format",
        choices=["jsonl", "txt"],
        default="jsonl",
        help="The storage format for the logs"
    )
    parser.add_argument(
        "-s", "--source",
        type=str,
        choices=['real', 'gen'],
        default='gen',
        help="source of samples: 'real' data, or 'generated' samples (default: 'gen')"
    )
    parser.add_argument(
        "-n", "--node",
        type=str,
        choices=['det', 'llm'],
        default='det',
        help="target-node: Detector or LLM_Agent (default: det)"
    )
    parser.add_argument(
        "-l", "--label",
        type=str,
        choices=['tn', 'fp', 'fn', 'tp'],
        default='tp',
        help="tn: True-Negative, fp: False-Positive, fn: False-Negative, tp: True-Positive (default: tp)"
    )
    parser.add_argument(
        "-t", "--test",
        type=int,
        #choices=range(1,11),
        default=1,
        help="test case index for selected targets: source, node, label( default: 1), -1 run all available samples!"
    )
    return parser.parse_args()