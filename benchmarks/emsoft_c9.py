from learn.cegis_barrier import Cegis
from utils.Config_B import CegisConfig
import timeit
import torch
from benchmarks.Exampler_B import get_example_by_name


def main():
    activations = ['SQUARE']  # Only "SQUARE","SKIP","MUL" are optional.
    hidden_neurons = [10] * len(activations)
    example = get_example_by_name('emsoft_c9')
    start = timeit.default_timer()
    opts = {
        "ACTIVATION": activations,
        "EXAMPLE": example,
        "N_HIDDEN_NEURONS": hidden_neurons,
        "MULTIPLICATOR": True,  # Whether to use multiplier.
        "MULTIPLICATOR_NET": [],  # The number of nodes in each layer of the multiplier network;
        # if set to empty, the multiplier is a trainable constant.
        "MULTIPLICATOR_ACT": [],  # The activation function of each layer of the multiplier network;
        # since the last layer does not require an activation function, the number is one less than MULTIPLICATOR_NET.
        "BATCH_SIZE": 2000,
        "LEARNING_RATE": 0.1,
        "MARGIN": 5.5,
        "LOSS_WEIGHT": (1.0, 1.0, 1.0),  # They are the weights of init loss, unsafe loss, and diffB loss.
        "SPLIT_D": True,  # Indicates whether to divide the region into 2^n small regions
        # when looking for negative examples, and each small region looks for negative examples separately.
        "DEG": [2, 2, 2, 2],  # Respectively represent the times of init, unsafe, diffB,
        # and unconstrained multipliers when verifying sos.
        "R_b": 0.7,
        "LEARNING_LOOPS": 100,
        "CHOICE": [0, 0, 0]  # For finding the negative example, whether to use the minimize function or the gurobi
        # solver to find the most value, 0 means to use the minimize function, 1 means to use the gurobi solver; the
        # three items correspond to init, unsafe, and diffB to find the most value. (note: the gurobi solver does not
        # supports three or more objective function optimizations.)
    }
    Config = CegisConfig(**opts)
    c = Cegis(Config)
    c.solve()
    end = timeit.default_timer()
    print('Elapsed Time: {}'.format(end - start))


if __name__ == '__main__':
    torch.manual_seed(167)
    main()
