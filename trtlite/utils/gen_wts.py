import pickle

import numpy as np


def convert(model_or_weight, dst_file, delete_str=None, flatten=True):
    if isinstance(model_or_weight, dict):
        ori_weights = model_or_weight
    else:
        ori_weights = model_or_weight.state_dict()

    weights = dict()
    with open(dst_file, 'w') as f:
        f.write('{}\n'.format(len(ori_weights.keys())))
        for k, v in ori_weights.items():
            if delete_str and k.startswith(delete_str):
                k = k.replace(delete_str, '')
            # if k in weights:
            #     continue
            if flatten:
                v = v.reshape(-1)
            vr = v.cpu().numpy()
            weights[k] = vr.astype(np.float32)

    with open(dst_file, 'wb') as f:
        pickle.dump(weights, f)
