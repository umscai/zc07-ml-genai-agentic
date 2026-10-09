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
from agents.agent_common import GraphNode
import sys

import time

tag_color_map={
GraphNode.detector: 'blue',
GraphNode.llm: 'yellow',
GraphNode.human: 'purple',
GraphNode.approved:'green',
GraphNode.blocked:'red',
GraphNode.dB:'cyan',
}
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
    node_id = args.node
    label_id=args.label
    test_id=args.test
    _ = CustomLogger(log_dir=settings.LOG_DIR, save_format=args.format,verbose=args.verbose)
    logger = logging.getLogger(__name__)
    print('Transaction Samples', tag='Get/Gen', tag_color='cyan', color='yellow')
    # if source_id == 'real':
    #     df = ut_test.get_random_sample_from_real_dataset(target_class= class_id, n_samples = num_samples)
    # else:
    #     df = ut_test.get_random_sample_from_generative_model(target_class = class_id, n_samples = num_samples)
    df = ut_test.get_test_case_sample(
        source_id = source_id,
        node_id = node_id,
        label_id = label_id,
        test_id = test_id
    )
    if df is None:
        print('Invalid test-id for selected source, node, label combination', tag='INVALID-INPUT', tag_color='red')
        print("Please see help for detail info using 'uv run agentic_system.py --help'", tag='See help Menu', tag_color='green')
        sys.exit(0)
    node = ut_test.node_map[node_id]
    label =ut_test.label_map[label_id]
    logger.info(f"Running test-id: {test_id} using {source_id} data with label:{label}, and target-node:{node}")

    #print(f"Running test-case-{test_case_id} for asset-id:{asset_id} with {num_samples} samples.")
    print(f"Test-id:{test_id}, Using {source_id} data with label:{label}, and target-node:{node}", tag='Running', tag_color='cyan', color='green')
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
                    for k,v in resp_dic.items():
                        print(f'{v}', tag=f'{k}-node', tag_color=tag_color_map[k])
                        if k == GraphNode.llm:
                            print('-'*60)
                            context = state.get('context',"") # note llm_context is summarized version of context
                            llm_reasoning = state.get('llm_reasoning',"")
                            llm_confidence = state.get('llm_confidence',0.0)
                            print(f':', tag='context', tag_color='yellow')
                            print(f'{context}')
                            print(f'{llm_reasoning}', tag='llm_reasoning', tag_color='yellow')
                            print(f'{llm_confidence}', tag='llm_confidence', tag_color='yellow')

    except Exception as e:
        logger.error(f"Error in assistant or main-telemetry loop: {e}")
        print(f"\nUnexpected error: {str(e)}", color='red')

#------------------------------------------------------------------------------
if __name__ == "__main__":

    main()
