""" Main function for the project"""
#------------------------------------------------------------------------------
import logging
from config import settings, CustomLogger, get_args
from print_color import print
import pprint
import utils as ut
from orchestration.assistant import FraudDetectionAssistant
from db_model.ltm_model import init_ltm_db
from db_model.transaction_model import init_trans_db
from orchestration.graph import init_checkpoint_db
import utils_test as ut_test

import time
#------------------------------------------------------------------------------
def main():
    args=get_args()
    print('Transaction', tag='Check-dB', tag_color='cyan', color='yellow')
    init_trans_db()
    print('LTM-Chroma-Vector-dB', tag='Check-dB', tag_color='cyan', color='yellow')
    init_ltm_db()
    print('Checkpoint', tag='Check-dB', tag_color='cyan', color='yellow')
    init_checkpoint_db()
    #print(f'{settings.USE_CONTEXT}', tag='Use LTM as Context', tag_color='cyan', color='red')
    print('+'*100)
    source_id = args.source
    class_id = args.class_id
    class_id_str = 'Fraud' if class_id == 1 else 'Legit'
    num_samples = args.num
    _ = CustomLogger(log_dir=settings.LOG_DIR, save_format=args.format,verbose=args.verbose)
    logger = logging.getLogger(__name__)
    print('Transaction Samples', tag='Get/Gen', tag_color='cyan', color='yellow')
    if source_id == 'real':
        df = ut_test.get_random_sample_from_real_dataset(target_class= class_id, n_samples = num_samples)
    else:
        df = ut_test.get_random_sample_from_generative_model(target_class = class_id, n_samples = num_samples)

    logger.info(f"Running test using {source_id} data with class-id:{class_id}({class_id_str}) and {num_samples} samples.")

    #print(f"Running test-case-{test_case_id} for asset-id:{asset_id} with {num_samples} samples.")
    print(f"Test: Using {source_id} data with class-id:{class_id}({class_id_str}) and {num_samples} samples.", tag='Running', tag_color='cyan', color='green')
    logger.debug({"msg": "init","status":"success"})
    try:
        assistant = FraudDetectionAssistant()
        for trans in df.to_dict(orient='records'):
            #time.sleep(.5) # make sure not faster than 1 sample per seconds!
            print('+'*100)
            trans_str=f"V1:{trans['V1']:.4f}, ..., V14:{trans['V14']:.4f}, ...,  V28:{trans['V28']:.4f}, , Amount:{trans['Amount']:.2f}, , Hour/2:{trans['Hour/2']}"
            print(f"{trans_str}", tag='TRANSACTION', tag_color='blue', color='yellow')
            result=assistant.run(source_id, trans)
            if result.get('success',False):
                state=result.get('state',{})
                #print(f"{state.get('response_dic')}")
                #pprint.pprint(f"{state.get('response_dic')}")
                resp_dic = state.get('response_dic',{})
                if resp_dic:
                    L= len(resp_dic)
                    if L < 3:
                        tc ='green'
                    elif L ==3:
                        tc='yellow'
                    else:
                        tc='red'
                    for k,v in resp_dic.items():
                        print(f'{v}', tag=f'{k}-node', tag_color=tc)
    except Exception as e:
        logger.error(f"Error in assistant or main-telemetry loop: {e}")
        print(f"\nUnexpected error: {str(e)}", color='red')

#------------------------------------------------------------------------------
if __name__ == "__main__":

    main()
