# Advancing Soft Robots with Evolutionary and Reinforcement Learning

**Lead Researcher**: [Aba Onumah](edin.com/in/aba-onumah-63315328b/)  
**Faculty Advisor**: Dr. Yu Xiang  
**Team Members**: Anish Reddy, Alle Arjun Krisnan, Jessica Myoungbin Choi, Rohit Penna

---

## Introduction and Purpose

Soft robotics is an emerging field that utilizes flexible materials such as rubber or EcoFlex to create robots capable of adapting to a wide variety of tasks. These robots are especially useful where traditional rigid robots may struggle, in dynamic environments like cave systems, disaster sites, agricultural fields, and deep-sea exploration.

However, designing and building soft robots from scratch to autonomously interact with these environments is a highly resource-intensive process, often requiring significant time, effort, and coordination to test and refine their performance. 

We present a unique approach by integrating **evolutionary algorithms (EAs)** with **reinforcement learning (RLs)**. This combination enables more efficient design and adaptation of soft robots, reducing both time and resource expenditures while enhancing their autonomy and robustness in unpredictable environments.

---

## Components

### Figure 1. Unity and SOFA Combined
| Image 1 | Image 2 |
| ------- | ------- |
| ![Alt Text 1](https://github.com/AbaDragon47/SoftEvo/blob/main/2024-11-15_6.05.43.png) | ![Alt Text 2](https://github.com/AbaDragon47/SoftEvo/blob/main/unity.png) |


For the development of our soft robotic model, we utilized two primary tools: **Unity** and the **SOFA Framework**.

- **Unity** provided the platform for simulating and visualizing the soft robot’s environment, leveraging its powerful physics engine and machine learning capabilities.
- **SOFA Framework** was employed to simulate soft body dynamics, enabling the modeling of deformable structures with high realism.

These tools were integrated by using Unity’s **ML-Agents** for environment simulation and SOFA’s components for the soft body physics. While we successfully managed to integrate deformations into the model, the robot’s movement was not fully realized at the time of this poster’s creation. Further research will focus on refining the movement dynamics and enhancing the interaction between components.

---

## Models

### **Proximal Policy Optimization (PPO)**
- A reinforcement learning (RL) algorithm that utilizes an **Actor-Critic Network**:
  - **Actor**: Determines the probabilities of taking certain actions from a given state.
  - **Critic**: Assesses an action by evaluating the current state.
- Ensures stability in policy updates by constraining changes within a threshold, avoiding drastic updates.

### **Deep Deterministic Policy Gradient (DDPG)**
- Designed to handle **continuous spaces** using an Actor-Critic model:
  - **Actor**: Learns a deterministic policy, mapping states to specific actions.
  - **Critic**: Estimates the value of actions using Q-learning.
- Utilizes a **replay buffer** for sampling past experiences and an **Ornstein-Uhlenbeck process** to introduce noise, encouraging exploration.

---

## Results

### **Figure 2**  
### **Figure 3**  

#### **Deep Deterministic Policy Gradient (DDPG)**
- Figure 1 illustrates the changing episodic reward of a DDPG model during training.
- A **replay buffer of state-action pairs** was used to avoid the need for infinite integration.
- Unity’s ML-Agents Python library simulated gymnasium environments, enabling visualization of training results.

#### **Proximal Policy Optimization (PPO)**
- Instead of evolving through hyperparameters, we crossed the weights of neural networks randomly between parent models to create the next generation.
- While successful in simple template environments, this method struggled with more complex environments simulated in Unity.

---

## Analysis

### **Figure 4**

By evolving populations of agents across multiple generations:
- EAs complemented RL by culling half the population, resulting in **higher learning efficiency** and **improved policy performance**.
- Techniques like **elitism** could further enhance training outcomes.

Results show that:
- DDPG’s EA-based approach effectively converged on suitable hyperparameters.
- PPO’s weight-crossing strategy performed well in simpler settings but struggled in complex environments.

---

## Conclusion

Preliminary findings suggest a promising trend:
- Hyperparameter tuning has led to an **exponential increase** in optimal rewards for both RL models and EAs.
- Continued research is likely to highlight the **effectiveness** of combining EAs and RLs in soft robotics, enhancing their performance in complex, dynamic environments.

---

## References

1. M. A. Graule, T. P. McCarthy, C. B. Teeple, J. Werfel, and R. J. Wood, "SoMoGym: A Toolkit for Developing and Evaluating Controllers and Reinforcement Learning Algorithms for Soft Robots," *IEEE Robotics and Automation Letters*, vol. 7, no. 2, pp. 4071-4078, April 2022.
2. Bai, Hui, et al. "Evolutionary Reinforcement Learning: A Survey." *Intelligent Computing*, vol. 2, 2023, pp. 1-10, doi:10.34133/icomputing.0025. [arXiv.org](https://arxiv.org/pdf/2303.04150).
3. Mertan, Alican, and Nick Cheney. "Modular Controllers Facilitate the Co-Optimization of Morphology and Control in Soft Robots." *Proceedings of the Genetic and Evolutionary Computation Conference, GECCO ’23*, 2023, pp. 174-183, doi:10.1145/3583131.3590416. [arXiv.org](https://arxiv.org/pdf/2306.09358v1).
