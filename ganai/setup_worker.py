import os
import sys
import json

def setup_worker(workers_addr:list[str], this_worker_idx:int):
    os.environ.pop("TF_CONFIG", None)
    
    if '.' not in sys.path:
        sys.path.insert(0, '.')
    
    tf_config = {
        'cluster': {
            'worker': workers_addr
        },
        'task': {
            'type': 'worker',
            'index': this_worker_idx
        }
    }
    
    os.environ['TF_CONFIG'] = json.dumps(tf_config)