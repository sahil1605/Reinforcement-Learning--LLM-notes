# Module 1: Sequential Decision Making in Reinforcement Learning

## Table of Contents
1. [Introduction to Sequential Decision Making](#introduction-to-sequential-decision-making)
2. [The Multi-Armed Bandit Problem](#the-multi-armed-bandit-problem)
3. [Action-Values and Expected Rewards](#action-values-and-expected-rewards)
4. [Exploration vs Exploitation](#exploration-vs-exploitation)
5. [Incremental Learning and Value Updates](#incremental-learning-and-value-updates)
6. [Exploration Strategies](#exploration-strategies)
7. [Practical Examples and Implementation](#practical-examples-and-implementation)
8. [Summary and Key Takeaways](#summary-and-key-takeaways)

---

## Introduction to Sequential Decision Making

### What is Sequential Decision Making?

Sequential decision making is the process of making a series of decisions over time, where each decision affects future outcomes and available options. In reinforcement learning, this is the core problem we're trying to solve.

**Key Characteristics:**
- **Temporal Nature**: Decisions are made over time, not all at once
- **Consequences**: Each action affects future states and rewards
- **Uncertainty**: We don't know the exact outcomes of our actions
- **Learning**: We improve our decision-making through experience

### The RL Framework

In reinforcement learning, we model sequential decision making through:

```
Agent → Action → Environment → State + Reward → Agent
```

**Components:**
- **Agent**: The decision maker
- **Environment**: The world in which the agent operates
- **State (s)**: Current situation/context
- **Action (a)**: Choice made by the agent
- **Reward (r)**: Feedback from the environment
- **Policy (π)**: Strategy for choosing actions

### Types of Sequential Decision Problems

1. **Multi-Armed Bandit**: Simplest case - no state information
2. **Contextual Bandits**: Actions depend on context/state
3. **Markov Decision Processes (MDPs)**: Full sequential decision making with states

---

## The Multi-Armed Bandit Problem

### Problem Definition

The multi-armed bandit is the simplest sequential decision problem where:
- An agent faces k different actions (arms)
- Each action has an unknown reward distribution
- Goal: Maximize cumulative reward over time

**Real-world Examples:**
- Clinical trials (treatments)
- Online advertising (ad selection)
- Website optimization (A/B testing)
- Resource allocation

### Mathematical Formulation

For k arms, each arm i has:
- **True value**: q*(a) = E[R|A=a]
- **Estimated value**: Q(a) ≈ q*(a)
- **Goal**: Find argmax_a Q(a)

### The Exploration-Exploitation Dilemma

This is the fundamental trade-off in sequential decision making:

- **Exploitation**: Use current best knowledge
- **Exploration**: Gather new information
- **Challenge**: Balance between immediate reward and long-term learning

---

## Action-Values and Expected Rewards

### Understanding Action-Values

An action-value Q(a) represents the expected reward for taking action a:

```
Q(a) = E[R|A=a] = Σ_r P(r|a) · r
```

Where:
- P(r|a): Probability of reward r given action a
- Q(a): Expected reward (action-value)
- R: Random variable representing reward

### Sample-Averaging Method

The simplest way to estimate action-values:

```
Q_n(a) = (R₁ + R₂ + ... + R_n) / n
```

**Properties:**
- Unbiased estimator
- Converges to true value as n → ∞
- Requires storing all past rewards

### Optimal Action Selection

The goal is to select the action that maximizes expected reward:

```
A_t = argmax_a Q(a)
```

**Challenges:**
- Q(a) is an estimate, not the true value
- Need to balance exploration and exploitation
- Uncertainty in estimates

---

## Exploration vs Exploitation

### The Fundamental Trade-off

**Exploitation:**
- Choose the action with highest estimated value
- Maximize immediate reward
- Risk: May miss better actions

**Exploration:**
- Try actions with uncertain values
- Gather information for future decisions
- Risk: May receive lower immediate rewards

### Sample-Averaging for Action Selection

We can measure how "successful" an action has been:

```
Q_n(a) = (Sum of rewards for action a) / (Number of times action a was chosen)
```

This gives us the average reward per action, helping us compare different actions.

### The Greedy Approach

**Greedy Action Selection:**
```
A_t = argmax_a Q_t(a)
```

**Problems:**
- No exploration
- May get stuck in local optima
- Doesn't learn about other actions

---

## Incremental Learning and Value Updates

### Why Incremental Learning?

Instead of storing all past rewards, we can update our estimates incrementally as new information arrives.

### Incremental Update Rule

**General Form:**
```
Q_{n+1} = Q_n + α[R_n - Q_n]
```

Where:
- Q_n: Previous estimate
- R_n: New observed reward
- α: Step-size parameter (learning rate)

### Understanding the Update

The update rule can be rewritten as:
```
Q_{n+1} = (1-α)Q_n + αR_n
```

This shows that the new estimate is a weighted average of:
- Old estimate: (1-α)Q_n
- New observation: αR_n

### Step-Size Parameters

**Fixed Step-Size (α = constant):**
- Recent rewards have more influence
- Good for non-stationary environments
- Never fully converges

**Decaying Step-Size (α = 1/n):**
```
Q_{n+1} = Q_n + (1/n)[R_n - Q_n]
```
- Equal weight to all observations
- Converges to true value
- Good for stationary environments

### Convergence Properties

**Sample-Averaging (α = 1/n):**
- Guaranteed convergence to true value
- Equal weight to all past rewards
- Best for stationary problems

**Constant Step-Size (α = constant):**
- Faster adaptation to changes
- Recent rewards have more influence
- Better for non-stationary problems

---

## Exploration Strategies

### 1. ε-Greedy Strategy

**Algorithm:**
```
A_t = {
    argmax_a Q_t(a)     with probability (1-ε)
    random action       with probability ε
}
```

**Parameters:**
- ε = 0: Pure exploitation (greedy)
- ε = 0.1: 10% exploration
- ε = 1: Pure exploration (random)

**Advantages:**
- Simple to implement
- Guarantees exploration
- Easy to tune

**Disadvantages:**
- Explores randomly, not intelligently
- May waste time on clearly bad actions

### 2. Upper Confidence Bound (UCB)

**Algorithm:**
```
A_t = argmax_a [Q_t(a) + c√(ln(t)/N_t(a))]
```

Where:
- Q_t(a): Estimated value of action a
- N_t(a): Number of times action a selected
- t: Total time steps
- c: Confidence level parameter

**Intuition:**
- First term: Exploitation (estimated value)
- Second term: Exploration (uncertainty bonus)
- Actions with high uncertainty get higher priority

**Advantages:**
- Intelligent exploration
- Theoretical guarantees
- Balances exploration and exploitation

**Disadvantages:**
- More complex to implement
- Requires tuning parameter c

### 3. Thompson Sampling (Bayesian Approach)

**Algorithm:**
1. Sample from posterior distribution of each action
2. Select action with highest sampled value

**Advantages:**
- Naturally balances exploration and exploitation
- Good theoretical properties
- Works well in practice

---

## Practical Examples and Implementation

### Example 1: 10-Armed Testbed

**Problem Setup:**
- 10 different actions (arms)
- Each arm has a different reward distribution
- Goal: Find the best arm over 1000 time steps

**Implementation Steps:**
1. Initialize Q(a) = 0 for all actions
2. For each time step:
   - Choose action using exploration strategy
   - Observe reward
   - Update Q(a) using incremental rule
3. Track performance metrics

### Example 2: Clinical Trial Simulation

**Scenario:**
- Testing 5 different treatments
- Each treatment has unknown effectiveness
- Goal: Maximize patient outcomes

**Key Considerations:**
- Ethical constraints (can't give clearly inferior treatment)
- Limited time horizon
- Non-stationary environment (treatments may change)

### Code Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class EpsilonGreedyBandit:
    def __init__(self, n_arms, epsilon=0.1, alpha=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.zeros(n_arms)  # Action values
        self.N = np.zeros(n_arms)  # Action counts
    
    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.Q)
    
    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += self.alpha * (reward - self.Q[action])
    
    def run_experiment(self, true_values, n_steps=1000):
        rewards = []
        optimal_actions = []
        
        for step in range(n_steps):
            action = self.select_action()
            reward = np.random.normal(true_values[action], 1)
            self.update(action, reward)
            
            rewards.append(reward)
            optimal_actions.append(action == np.argmax(true_values))
        
        return rewards, optimal_actions

# Example usage
true_values = np.random.normal(0, 1, 10)
bandit = EpsilonGreedyBandit(n_arms=10, epsilon=0.1)
rewards, optimal = bandit.run_experiment(true_values)
```

### Performance Metrics

**Key Metrics to Track:**
1. **Average Reward**: Cumulative reward over time
2. **Optimal Action Percentage**: How often the best action is chosen
3. **Regret**: Difference between optimal and actual performance
4. **Convergence Rate**: How quickly the algorithm learns

---

## Summary and Key Takeaways

### Core Concepts

1. **Sequential Decision Making**: Making decisions over time with consequences
2. **Action-Values**: Expected rewards for different actions
3. **Exploration vs Exploitation**: Fundamental trade-off in learning
4. **Incremental Learning**: Updating estimates with new information

### Key Algorithms

1. **ε-Greedy**: Simple, effective exploration strategy
2. **UCB**: Intelligent exploration based on uncertainty
3. **Thompson Sampling**: Bayesian approach to exploration

### Practical Considerations

1. **Step-Size Selection**: Balance between learning speed and stability
2. **Exploration Rate**: Tune based on problem characteristics
3. **Performance Metrics**: Track multiple measures of success
4. **Non-Stationarity**: Adapt to changing environments

### Next Steps

This module provides the foundation for understanding sequential decision making. The next modules will cover:
- Markov Decision Processes (MDPs)
- Value Functions and Policies
- Dynamic Programming
- Monte Carlo Methods
- Temporal Difference Learning

### Exercises

1. Implement and compare ε-greedy vs UCB on a 10-armed bandit
2. Experiment with different step-sizes and their effects
3. Analyze the exploration-exploitation trade-off
4. Apply bandit algorithms to a real-world problem

---

*This module establishes the fundamental concepts of sequential decision making in reinforcement learning, providing the foundation for more advanced topics in subsequent modules.*
