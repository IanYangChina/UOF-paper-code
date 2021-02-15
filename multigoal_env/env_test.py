import numpy as np
from gym import make
import matplotlib.pyplot as plt
import multigoal_env
np.set_printoptions(precision=2)
env = make('TwoObjectRandomSize-v0')
ind = 0
for i in range(100):
    env.reset()
    env.set_goals(sub_goal_ind=ind)
    time_done = False
    while not time_done:
        # a = env.action_space.sample()
        a = np.array([0.0, -0.0, 0.0, 0.0])
        obs_, act_reward, time_done, act_info = env.step(a)
        # using sim.render needs to empty the LD_PRELOAD path and execute the script on terminal
        # export LD_PRELOAD=
        # img = env.env.sim.render(width=400, height=400, depth=True)
        # plt.imshow(img[1], cmap='Greys')
        # plt.show()
        # env.render()
    ind = (ind+1) % len(env.sub_goal_space)