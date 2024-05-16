This is the repository for my project for CISC 889 Deep Learning.


The goal with this project was to make a patient agent that would wait to achieve a more maximal reward.
This is done through the introduction of a discount function, as well as the notion of time into a traditional Q-learning model.
This new discount function is a piecewise function of 2 constants, such that before a set time T it will output a discount greater than 1, and after this time will output a discount lower than 1, closer to a traditional discount value.
In turn, the agent will be encouraged to wait until time T to reach the goal.
The agent must have a concept of time, or at least the Q-values should, which requires adding a new dimension onto the Q-value table.  Primarily, this hurts the space efficiency and can lead to a lot of sparsity in the table.
In all, this method requires the tuning of a lot of hyperparameters, but is able to efficiently create a patient agent for small problem sizes.


Other interesting avenues to explore related to this topic include other discount functions as well as multi-goal environment.
The specific discount function chosen was to promote arrival at a set time, however any monotonic never increasing function could be used instead.
Functions like exponential decay would be interesting to see here, though they would not accomplish the exact same goal.
If there is an environment with, for example, 2 goals and the agent has to reach one goal before the other, this discount implementation could be used for the later goal.
One other method to cut down on runtime, space efficiency, as well as sparsity in the Q-value table would be to reduce the size of the time dimension to size T+1, where the first T times are all distinct, and all times after T are equivalent.
This would result in the agent being encouraged to, if it can't reach the goal in time, use the traditional Q-learning to just reach the goal at some point.



Additional comments:
* The primary code is in main_large_table.py.
* Runtime is poor, because I was unable to get CUDA working, otherwise main_gpu.py would be used instead.  If I could have gotten CUDA working I would have been able to train much larger models.
* Most of the important changes are in the class DelayedReward.  Other hyperparameters as well as the changes to the Q-value table are included outside this class.
* Sample gifs and reward graphs from different trainings are found in ./log/
* A sample training/running example is given in ./project_video.mp4
