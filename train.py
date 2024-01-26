import os
from argparse import ArgumentParser

from learner.arguments import parser
from learner.dmc import train

if __name__ == '__main__':
    flags = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    flags.load_model = True

#    flags.actor_device_cpu=True
    flags.no_gpu = False
#    flags.training_device = 'cpu'

    assert flags.load_model == True
    train(flags)
