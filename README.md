### Official implementation for the UOF paper (algorithm &amp; environment)

This is the official implementation of the codes that produced the results in the 2021 IEEE TNNLS paper titled 
**"Hierarchical Reinforcement Learning with Universal Policies for Multi-Step Robotic Manipulation"**.
Feel free to play with the codes and raise issues.

<img src="/src/graphical_abstract.jpg" width=1080>

#### Citation: to be updated.

#### Main Dependencies:
1. Ubuntu 16.04
    - Higher version Ubuntu systems should work as well.
    - The project was developed and tested on Linux, not sure how it works on Windows.
2. Python 3
3. [Mujoco150](https://www.roboti.us/index.html)
4. [Mujoco-py==1.50.1.68](https://github.com/openai/mujoco-py/tree/master) `python -m pip install mujoco-py==1.50.1.68`
5. [Pytorch](https://pytorch.org/get-started/locally/)
6. Others `python -m pip install -r requirements.txt`

#### Get started:
1. Clone the repository to wherever you like.
2. On a terminal: `export PYTHONPATH=$PYTHONPATH:$PATH_OF_THE_PROJECT_ROOT`. Replace `$PATH_OF_THE_PROJECT_ROOT` with
something like `/home/someone/UOF-paper-code`.
3. (Optional) Activate your conda environment if desired.
4. From the project root: 
    - Evaluate the pre-trained UOF agent `python run_uof.py --task-id 0`
    - Evaluate the pre-trained HAC agent `python run_hac.py --task-id 0`
    - Train your own UOF agent `python run_uof.py --task-id 0 --train`
    - Train your own HAC agent `python run_hac.py --task-id 0 --train`

#### Task id (pre-trained policy) - paper result relation

This table gives the correspondence between the pre-trained policies provided in this repo and the performance
given in the paper figures. The given UOF policies were trained with AAES and 0.75 demonstration proportion.

| | |
| :------ | :---------------------------- |
| Task id | Paper result (red curves)     |
| 0       | Fig. 4a, 5a, 9a, 9c, 10a      |
| 1       | Fig. 5b, 6, 7, 8, 9b, 9d, 10b |
| 2       | Fig. 10c                      |
| 3       | Fig. 10d                      |
| 4       | Section VII-G                 |
| 5       | Section VII-G                 |
| 6       | Section VII-G                 |
| 7       | Section VII-G                 |

#### Full argument list:

`run_uof.py`

| | |
| :---------------------- | :----------------------------------------------- |
| Arguments               | Description                                      |
| `--task-id i`           | Task id, where, $i \in {0, 1, 2, ..., 7}$        |
| `--render`              | Use this flag if you want to render the task     |
| `--train`               | Use this flag for training an agent from scratch |
| `--multi-inter`         | Use this flag to train separate high-level policies for each goal |
| `--no-aaes`             | Use this flag to turn off the AAES exploration strategy |
| `--no-demo`             | Use this flag to turn off the Abstract Demonstrations |
| `--demo-proportion j`   | Use this flag to set the proportion of episodes that use demonstrations, where, $j \in {0.0, 0.25, 0.5, 0.75, 1.0}$ |

For `run_hac.py`, ignore the `--multi-inter` and `--no-aaes` arguments.