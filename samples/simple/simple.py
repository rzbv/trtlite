import os
import pickle
import tempfile

import numpy as np

import trtlite
import trtlite.nn as nn
import trtlite.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def test():
    # create template weights
    model_dict = {}
    model_dict['fc1.weight'] = np.random.randn(20, 10).astype(np.float32)
    model_dict['fc1.bias'] = np.random.randn(20).astype(np.float32)
    model_dict['fc2.weight'] = np.random.randn(5, 20).astype(np.float32)
    model_dict['fc2.bias'] = np.random.randn(5).astype(np.float32)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine_file = os.path.join(tmp_dir, 'model.plan')
        weights_file = os.path.join(tmp_dir, 'model.pkl')

        with open(weights_file, 'wb') as f:
            pickle.dump(model_dict, f)
    
        # Instantiate the network
        model = SimpleNet()

        # Build the TensorRT engine
        config = {
            'engine_file': engine_file,
            'weight_file': weights_file,
            'precision': 'fp16',
            'inputs': [('x', (-1, 1, 1, 10))],
            'input_profiles': [((1, 1, 1, 10), (16, 1, 1, 10), (32, 1, 1, 10))],
        }
        model.build_engine(config)

        # Run inference
        session = trtlite.InferenceSession(engine_file)
        x = np.random.rand(16, 1, 1, 10).astype(np.float32)
        output = session.run({'x': x})
        print("Output:", output['output_0'].shape)


if __name__ == '__main__':
    test()