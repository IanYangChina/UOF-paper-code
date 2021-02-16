### Official implementation for the UOF paper (algorithm &amp; environment)

#### Status: under reconstruction, don't try to use it for now

This is the official implementation of the codes that produced the results in the 2021 IEEE TNNLS paper titled 
**"Hierarchical Reinforcement Learning with Universal Policies for Multi-Step Robotic Manipulation"**.

<img src="/src/graphical_abstract.jpg" width=1080>

#### Main Dependencies:
1. Ubuntu 16.04
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
4. From the project root: `python run.py --env '0' --agent 'UOF'`

#### Full argument list:

to be filled