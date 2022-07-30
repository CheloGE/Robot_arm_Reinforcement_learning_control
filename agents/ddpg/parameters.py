BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 4096       # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 1e-9     # L2 weight decay