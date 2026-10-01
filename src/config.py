"""Project config file"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from pydantic import computed_field
import numpy as np

import logging
import json
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
import argparse

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


    RANDOM_SEED:int = 6488
    RNG: np.random.Generator = np.random.default_rng(RANDOM_SEED)

    MODEL_NAME: str   #=  'MODEL_NAME' # will load MODEL_NAME from .env
    OPENAI_API_KEY: str #= 'OPENAI_API_KEY' # will load OPENAI_API_KEY from .env
    OPENAI_BASE_URL: str # # will load OPENAI_API_KEY from .env
    TEMPERATURE: float # = 0.0 # will load TEMPERATURE from .env and convert to float
    USE_CONTEXT:bool #will load USE_CONTEXT from .env and convert to boolean
    #
    TRANSACTION_DB:str = 'transaction.db'
    LTM_DB:str = 'ltm.db'
    CHECKPOINT_DB:str = 'checkpoint_db'
    WORKFLOW_GRAPH: str = 'workflow_graph.png'
    #
    HOURS_IN_TIME_GROUP:int = 2 # after converting time to Hours in [0,23], we group them together [0,1]->0, [2,3]-> 1,...


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

#------------------------------------------------------------------------------
# Create a single instance to be used everywhere
settings = Settings()
settings.create_directories()
#settings.check_all_db_exist()


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
def get_args():
    """
    Handles command-line arguments.
    Easily expandable by adding more parser.add_argument() calls.
    """
    parser = argparse.ArgumentParser(description="Arg Parser")

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
    # parser.add_argument(
    #     "-a", "--asset",
    #     type=int,
    #     choices=[1, 2, 3],
    #     default=1,
    #     help="asset-id/index to get normal operational parameters,...(default: 1)"
    # )
    # parser.add_argument(
    #     "-t", "--test",
    #     type=int,
    #     choices=[1, 2, 3,4],
    #     default=1,
    #     help="Test-Case number (default: 1)"
    # )

    # parser.add_argument(
    #     "-s", "--samples",
    #     type=int,
    #     default=1,
    #     help="Number of telemetry samples to generate and inject to system(default: 1)"
    # )


    return parser.parse_args()