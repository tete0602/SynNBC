# 1.Introduction

SynNBC is a new software toolbox for safety verification based on neural barrier certificates synthesis for continuous dynamical systems. We construct the synthesis framework as an inductive loop between a Learner and a Verifier based on barrier certificate learning and counterexample guidance. In the counterexample generation phase,we use the special form to convert the counterexample generation into a polynomial optimization problem for obtaining the optimal counterexample. In the verification phase, the task of identifying the real barrier certificate can be tackled by solving the Linear Matrix Inequalities (LMI) feasibility problem.

The directory in which you install SynNBC contains five subdirectories:

* `/benchmarks`: the source code and some examples;
* `/learn`: the code of learners;
* `/verify`: the code of verifiers;
* `/plots`: the code of plots;
* `/utils`: the configuration of CEGIS.

# 2.Configuration

## 2.1 System requirements

To install and run SynNBC, you need:

* Windows Platform: `Python 3.9.12`;
* Linux Platform: `Python 3.9.12`;
* Mac OS X Platform: `Python 3.9.12`.

## 2.2 Installation instruction

You need install required software packages listed below and setting up a MOSEK license .

1. Download SynNBC.zip, and unpack it;
2. Install the required software packages for using SynNBC:

    ```python
    pip intsall matplotlib==3.5.3
    pip intsall numpy==1.23.2
    pip intsall scipy==1.9.0
    pip intsall SumOfSquares==1.2.1
    pip intsall sympy==1.11
    pip intsall torch==1.12.1
    pip install Mosek==10.0.30
    pip install gurobipy==10.0.0
    pip install picos==2.4.11
    ```

3. Obtain a fully featured Trial License if you are from a private or public company, or Academic License if you are a student/professor at a university.

* Free licenses
  * To obtain a trial license go to <https://www.mosek.com/products/trial/>
  * To obtain a personal academic license go to <https://www.mosek.com/products/academic-licenses/>
  * To obtain an institutional academic license go to <https://www.mosek.com/products/academic-licenses/>
  * If you have a custom license go to <https://www.mosek.com/license/request/custom/> and enter the code you received.
* Commercial licenses
  * Assuming you purchased a product ( <https://www.mosek.com/sales/order/>) you will obtain a license file.

# 3.Automated Synthesis of Barrier Functions

Main steps to generate verified barrier functions:

1. Create a new example and confirm its number;
2. Input dimension `n`, three domains: `D_zones,I_zones and U_zones` and differential equations `f`;
3. Define the example’s name, call function `get_example_by_name`, input parameters of opts and get verified barrier functions.

## 3.1 New examples

In SynNBC, if we want to generate a barrier function, at first we need create a new example in the examples dictionary in `Exampler_B.py`. Then we should confirm its number. In an example, its number is the key and value is the new example constructed by Example function.

```python
>>  1 : Example ()
```

## 3.2 Inputs for new examples

At first, we should confirm the dimension `n` and three domains: `D_zones,I_zones and U_zones`. For each domain, the number of the ranges must match the dimension `n` input.

**Example 1** &emsp; Suppose we wish to input the following domains:

$$D\_zones = \{-2 \leq x_1 \leq 2 ,-2 \leq x_2 \leq 2\}$$
$$I\_zones = \{0 \leq x_1 \leq 1 ,1 \leq x_2 \leq 2\}$$
$$U\_zones = \{-2 \leq x_1 \leq -0.5 ,-0.75 \leq x_2 \leq 0.75\}$$

This can be instantiated as follows:

```python
>>  n=2,
>>  D_zones=[[-2, 2]] * 2,
>>  I_zones=[[0, 1], [1, 2]],
>>  U_zones=[[-2, -0.5], [-0.75, 0.75]],
```

Then, the dynamical system should be confirmed in the Example function. The dynamical system is modelled as differential equations `f`. We define the differential equations through lambda expressions. The variables $x_1,x_2,x_3,\cdots,x_n$ should be typed as $x[0], x[1], x[2], \cdots, x[n-1]$. All differential equations are input into the *f* list.

For Example 1, we consider the following differential equations:

$$f_1 = x_2 + 2*x_1*x_2$$
$$f_2 = -x_0 - x_1 ^ 2 + 2* x_0 ^ 2$$

Construct the differential equations by setting

```python
>>  f=[
        lambda x: x[1] + 2 * x[0] * x[1],
        lambda x: -x[0] - x[1] ** 2 + 2 * x[0] ** 2
    ],
```

## 3.3 Getting barrier functions

After inputting the dimension, domains and `f`, we should define the example’s name. For instance, to create an example named `barr_1`, you need type:

```python
>> name = 'barr_1'
```

The completed example is following:

```python
>> 2: Example(
        n=2,
        D_zones=[[-2, 2]] * 2,
        I_zones=[[0, 1], [1, 2]],
        U_zones=[[-2, -0.5], [-0.75, 0.75]],
        f=[
            lambda x: x[1] + 2 * x[0] * x[1],
            lambda x: -x[0] - x[1] ** 2 + 2 * x[0] ** 2
        ],
        name='barr_1'
    ),
```

Then we should update the code of `barrier_template.py` or create a new python file by imitating its code. For generating a barrier function,we should input the parameter name to call function `get_example_by_name` and set the parameters of opts.

For Example 1, the code example is as follows:

```python
>>  activations = ['SKIP']  # Only "SQUARE","SKIP","MUL" are optional.
>>  hidden_neurons = [10] * len(activations)
>>  example = get_example_by_name('barr_1')
>>  opts = {
        "ACTIVATION": activations,
        "EXAMPLE": example,
        "N_HIDDEN_NEURONS": hidden_neurons,
        "MULTIPLICATOR": True,  # Whether to use multiplier.
        "MULTIPLICATOR_NET": [],  # The number of nodes in each layer of the multiplier network;
        # if set to empty, the multiplier is a trainable constant.
        "MULTIPLICATOR_ACT": [],  # The activation function of each layer of the multiplier network;
        # since the last layer does not require an activation function, the number is one less than MULTIPLICATOR_NET.
        "BATCH_SIZE": 500,
        "LEARNING_RATE": 0.1,
        "LOSS_WEIGHT": (1.0, 1.0, 1.0),  # They are the weights of init loss, unsafe loss, and diffB loss.
        "MARGIN": 0.5,
        "SPLIT_D": not True,  # Indicates whether to divide the region into 2^n small regions
        # when looking for negative examples, and each small region looks for negative examples separately.
        "DEG": [2, 2, 2, 1],  # Respectively represent the times of init, unsafe, diffB,
        # and unconstrained multipliers when verifying sos.
        "R_b": 0.6,
        "LEARNING_LOOPS": 100,
        "CHOICE": [0, 0, 0]  # For finding the negative example, whether to use the minimize function or the gurobi
        # solver to find the most value, 0 means to use the minimize function, 1 means to use the gurobi solver; the
        # three items correspond to init, unsafe, and diffB to find the most value. (note: the gurobi solver does not
        # supports three or more objective function optimizations.)
    }
```

At last, run the current file and we can get verified barrier functions. For Example 1, the result is as follows:

```python
B = -0.793783350902763*x1**2 - 0.196982929294886*x1*x2 - 2.04443478700998*x1 + 1.1180428026877*x2**2 - 3.80881503168995*x2 + 2.51228422783734
```

At the same time, if the dimension `n` is 2, then a three-dimensional image of the `Barrier Certificate` and a two-dimensional image of the `Barrier Border` will be generated.For example 1, the image is as follows:

![Barrier Certificate](https://github.com/tete0602/SynNBC/blob/main/benchmarks/img/Barr1_3d.png)
![Barrier Border](https://github.com/tete0602/SynNBC/blob/main/benchmarks/img/Barr1_2d.png)
