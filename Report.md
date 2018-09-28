[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Networks
* Deep neural network - [network.py](network.py): a 3-layer full connected layer with ReLU in between.
```
37 Input -> 64 FC, ReLU -> 64 FC, ReLU -> 4 FC
```
* Dueling Network - [dueling_network.py](dueling_network.py): a feature layer connected to both state value layer and action advantage layer. The outputs from state value layer and advantage layer are combined to produce the final Q-value. Dueling Network is made specifically for q-learning. The network emphasizes the fact that in some states there is not much difference among available actions while in other states there is a big difference among available actions. The detailed information about the network can be found from [the paper](https://arxiv.org/abs/1511.06581), 'Dueling Network Architectures for Deep Reinforcement Learning'.
```
37 Input -> 64 FC, ReLU -> 64 FC, ReLU -> 1 FC |
                                               |+ -> 4
                        -> 64 FC, ReLU -> 4 FC |
```
And the formula for the aggregation layer is:
$$
Q(s,a;\theta,\alpha,\beta) = V(s;\theta,\beta) 
+ \bigg(A(s,a;\theta,\alpha) - \frac{1}{|\mathcal{A}|}\sum_{a'}A(s,a';\theta,\alpha) \bigg)
$$

### Agent
#### Q-Network Agent
Q-Network agent, [dqn_agent.py](dqn_agent.py), implements the Deep Q-Learning algorithm from [DQN paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). 

1. Uses experience replay buffer to learn more efficiently from past experience, instead of throwing away. It also makes the agent more robust by breaking the correlation among experience sequence.
2. Employs two networks to stablize the learning process. A behavior network with parameter $\theta$ determines the policy and updates at every step, while a target network with parameter $\theta'$ is only updated periodically from $\theta$. It provides a stable target for a while for learning before changing to the next target. The formula of DQN target is updated as follows:
    $$
    Y_t^{DQN} = r_{t+1} + \gamma \max_a Q(s_{t+1},a,\theta_t')
    $$
    $$
    \theta_{t+1} = \theta_t + \alpha \big(Y_t^{DQN} - Q(s_t,a_t;\theta_t)\big) \nabla_{\theta_t} Q(s_t,a_t;\theta_t)
    $$
   
#### Double Q-Network Agent
Double Q-Network Agent, [ddqn_agent.py](ddqn_agent.py), implements the improvements from [the paper](https://arxiv.org/abs/1509.06461), 'Deep Reinforcement Learning with Double Q-learning'.
1. To reduce overestimations of Q-Network Agent, we can introduce a simple cross checking between the two networks, behavior network and target network, from Q-Network Agent. We can get the two networks to both agree the chosen best action. The DQN target is updated as follows, the behavior network pick the next best action and the target network determines the action value of the chosen best action: 
$$
Y_t^{DoubleDQN} = r_{t+1} + \gamma Q\big(s_{t+1},argmax_a Q(s_{t+1}, a; \theta_t);\theta_t')
$$


#### Prioritized Experience Replay
Researchers at DeepMind improved the previous experience replay of sampling uniformly in the [paper](https://arxiv.org/abs/1511.05952), 'Prioritized Experience Replay'. 
1. Instead of treat every experience the same, we expect to learn more from the experiences having larger TD error. So we can define the priority $p$ of experience $i$ as TD-error $\delta$ plus a small number $\epsilon$(to avoid 0 priority). Alpha, $0 <= \alpha <= 1$, controls how much prioritization is used. With $\alpha = 0$, we get uniform sampling. 
    $$
    p_i = (|\delta_i| + \epsilon)^\alpha
    $$
    The probability $P(i)$ of an experience $i$ being chosen is its priority divided by the sum of all $N$ priorities in the replay buffer.

$$
P(i) = \frac{p_i}{\sum_k p_k}
$$

2. To improve sampling efficiency, a sum tree structure is implemented in [sum_tree.py](sum_tree.py), which references the implementation from [Jaromir's blog post](https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/).
3. Prioritized experience replay introduces bias because it produces sample batch in an uncontrolled fashion. We correct this bias by using importance-sampling weights.
    $$
    w(i) = \bigg(\frac{1}{N} \cdot \frac{1}{P(i)}\bigg)^\beta
    $$
    we use $\beta$ to control how much we compensate for the non-uniform probabilities $P(i)$, full compensation if $\beta=1$. In typical reinforcement learning scenarios, the unbiased nature of the updates is most important near convergence at the end of training. So we can linearly anneal $\beta$ from a initial value(I adopt 0.6 used in the paper) to 1 during training. The relevant files are [prioritized_replay_buffer.py](prioritized_replay_buffer.py) and [ddqn_prioritized_agent.py](ddqn_prioritized_agent.py).

#### Hyper-Parameters
* Replay buffer size, BUFFER_SIZE, of 100,000.
* Minibatch size, BATCH_SIZE, of 64.
* Discount factor, GAMMA, of 0.98
* Soft update of target parameters, TAU, of 0.001
* Learning rate, LR, of 0.0005
* How often to update the target network, UPDATE_EVERY, of 4.
* Small amount to avoid zero priority in prioritized replay buffer, epsilon, of 0.01.
* [0~1] determines how much prioritization is used in prioritized replay buffer, alpha, of 0.6.
* To control importance-sampling compensation, beta, of initial value 0.4.

#### Result
1. DQN training result with checkpoint saved to [dqn.pth](dqn.pth):
![DQN](/assets/dqn.jpg)
2. Double Q-Network training result with checkpoint saved to [ddqn.pth](ddqn.pth):
![DQN](/assets/ddqn.jpg)
3. Dueling Network training result with checkpoint saved to [dueling_network.pth](dueling_network.pth):
![DQN](/assets/dueling_network.jpg)
1. Dueling Network with Prioritized Experience Replay training result, and checkpoint saved to [dueling_network_per.pth](dueling_network_per.pth):
![DQN](/assets/dueling_network_per.jpg)


### Future Improvements
Due to time contraints, I have not been able to make a good comparison of the each approaches with good number of runs. Also I would like to experiment with Pixels input environment with CNN network in future.
