# SoftEvo - PPO Research

### Types of Reinforcement Learning

- **Value-based**: Selects actions that maximize the predicted reward value for the current state.
- **Policy-based**: Directly learns a **policy** that determines actions to take.
    - **Policy**: A function that decides which action to take in a given state.

### **Core Concepts of PPO (Proximal Policy Optimization)**

**Objective**: To update the policy without making too large a change from the previous policy, aiming for stability during the training process.

- **Clipping**: Used by PPO, to ensure that updates to the policy don’t result in drastic changes
  - Calculates the ratio between the new policy and the previous policy and restricts the update if the ratio changes too much (to ensure stable learning without sudden shifts in performance)
    
- **Objective Function**: A loss function that considers both the **reward** and the **extent of policy change**. Evaluates how much to improve the policy while ensuring that the update is not too extreme.
    
`PPO learns a new policy to increase rewards but ensures that the difference is not too large for stability.`
    

### **Why PPO is Effective**

- Relatively simple while maintaining stable training
- Can update the policy efficiently, even after multiple iterations, **keeping computational costs relatively low**
