
import json
from collections import OrderedDict

def load_params(param_file=None, verbose=True):
    # Parameters from params.json and exp.json loaded here to override parameters set below
    if param_file is not None:
        hyper_p = json.loads(open(param_file).read(),
                             object_pairs_hook=OrderedDict)
        if verbose:
            print('Using hyper-parameters:')
            for key, value in hyper_p.items():
                print('{:25s} - {:12}'.format(key, str(value)))
            print('rest of parameters are set as default')
        parameters = {
    # for starting model from a checkpoint
    'reload_model': False,
    'prev_epochs': 0,
    "verbose_print": 0,}

        parameters.update(hyper_p)
    return parameters