# SoftEvo Research - Deep Deterministic Policy Gradients

Matches regular state => action mapping of most policy solutions, but rather than discrete actions, create an action gradient
Since this is technically impossible we use a time delayed state to generate a slightly different action than the actual result would be.

state => action (determined from Q value), turns into
- state => action => (state, action*) => Q value
- The Q value is an evaluation of the return of an action using different reward functions for each state action set. argmax(Q) over a is the optimal action essentially (but you can't do that cuz that requires a finite amount of actions)
- Example Q: reward + discount (based on how far into the future the state is) * Q_next

The list of states will be represent as a stack of states (tuples of (state, action, reward, next_state, terminate_boolean)) and used to calculate the Q values for each subsequent state and action.