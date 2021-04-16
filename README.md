##Deep Q-Learning Architecture
Here we describe the structure of deep q learning so that we are able to optimize our learning model.

######Considerations
We begin by considering how we want our system to look like, that is since we are going to have an OOP
layout we will want to define the class considerations that we want to have and the functional reasons
behind the definition of the classes.
* So we know the context of our application and that we expect our learning algorithm to trade on a
  market and earn money. The learning algorithm will receive an input state which will be defined by
  the last 12 candle data from the market charts. The output action will be one of three decisions to
  make; that is buy, sell, or hold.
* We then apply the action to the environment. Since we want the result of the action and the state
  of the environment after, we will need to store get the reward from the new state.
* We then need to store the initial state, the approximated action, its result state and reward.

######Classes

|Name|Description|
|:-----|:-------|
|```Account```|*This class we will use to manage the actions and rewards of the agent, that is, we will trigger actions on the market from this class and record the position. We will also close positions to evaluate the reward. When closing the position we will need to supply the new state of the market.*|
|```Memory```|*This class we will use to cache the ```(state, action, reward, next)``` as well as manage the data that accumulates in the memory of the application.*|
|```Experience```|*We will model the ```(state, action, reward, next)``` data into an Experience class.*|
|```Agent```|*We need to model the actual agent so we will have this class to keep the DNN architecture, and the hyper-parameters that it will be working with.*|
|```Environment```|*We will also need to model the environment properly, which we have not done properly yet. What this means is that we will need to structure a place that will simulate the market movement so that the agent is able to act in the environment.*|
|```Position```|*We will define this class to represent the position responded from the environment.*|

######Pseudo Algorithm
1. Initialize Memory Capacity
2. Initialize the policy network with random weights.
3. Clone the policy network, and call it the target network.
4. For each episode,
    1.  Initialize the starting state.
        *   Via exploration or exploitation.
    2.  Execute selected action in an emulator or environment.
    3.  Observe reward and next state.
    4.  Store experience in memory.
    5.  Sample random batch from memory.
    6.  Preprocess state from batch.
    7.  Pass batch of preprocessed states to policy network.
    8.  Calculate loss between output Q-values and the target Q-values.
        *   Requires a pass to the target network for the next state.
    9. Gradient Descent updates weights in the policy network to minimize loss.
        *   After ```x``` time steps, weights in the target network are updated to the weights in the
        policy network.