# Fundamentals of Reinforcement Learning

## Table of Contents
1. [Introduction to Reinforcement Learning](#introduction-to-reinforcement-learning)
2. [Module 1: Sequential Decision Making](#module-1-sequential-decision-making)
3. [Key Concepts and Terminology](#key-concepts-and-terminology)
4. [The RL Learning Loop](#the-rl-learning-loop)
5. [Types of RL Problems](#types-of-rl-problems)
6. [Mathematical Foundations](#mathematical-foundations)
7. [Learning Objectives](#learning-objectives)

---

## Introduction to Reinforcement Learning

### What is Reinforcement Learning?

Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. The agent receives feedback in the form of rewards or penalties, allowing it to learn optimal behavior through trial and error.

### Key Characteristics of RL

- **Learning through Interaction**: Agent learns by acting in the environment
- **Trial and Error**: No explicit supervision, learns from consequences
- **Delayed Feedback**: Rewards may come after a sequence of actions
- **Sequential Decision Making**: Current actions affect future outcomes
- **Exploration vs Exploitation**: Balance between trying new things and using known good strategies

### The RL Framework

```
┌─────────┐    Action    ┌─────────────┐    State + Reward    ┌─────────┐
│  Agent  │ ──────────→  │ Environment │ ──────────────────→  │  Agent  │
└─────────┘              └─────────────┘                      └─────────┘
```

**Components:**
- **Agent**: The learner/decision maker
- **Environment**: The world the agent interacts with
- **State (s)**: Current situation/context
- **Action (a)**: Choice made by the agent
- **Reward (r)**: Numerical feedback signal
- **Policy (π)**: Strategy for choosing actions

---

## Module 1: Sequential Decision Making

### Introduction to Sequential Decision Making

Sequential decision making is the process of making a series of decisions over time, where each decision affects future outcomes and available options. In reinforcement learning, this is the core problem we're trying to solve.

**Key Characteristics:**
- **Temporal Nature**: Decisions are made over time, not all at once
- **Consequences**: Each action affects future states and rewards
- **Uncertainty**: We don't know the exact outcomes of our actions
- **Learning**: We improve our decision-making through experience

### The Multi-Armed Bandit Problem

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

### Action-Values and Expected Rewards

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

### The Exploration-Exploitation Dilemma

This is the fundamental trade-off in sequential decision making:

- **Exploitation**: Use current best knowledge
- **Exploration**: Gather new information
- **Challenge**: Balance between immediate reward and long-term learning

### Incremental Learning and Value Updates

Instead of storing all past rewards, we can update our estimates incrementally as new information arrives.

**Incremental Update Rule:**
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

### Exploration Strategies

#### 1. ε-Greedy Strategy

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

#### 2. Upper Confidence Bound (UCB)

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

#### 3. Thompson Sampling (Bayesian Approach)

**Algorithm:**
1. Sample from posterior distribution of each action
2. Select action with highest sampled value

**Advantages:**
- Naturally balances exploration and exploitation
- Good theoretical properties
- Works well in practice

### Practical Implementation Example

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

## Key Concepts and Terminology

### Essential RL Terms

**State (s)**: A representation of the current situation
- Can be fully observable or partially observable
- May include agent's internal state and environment state

**Action (a)**: A choice available to the agent
- Can be discrete (finite set) or continuous
- May have constraints or requirements

**Reward (r)**: Numerical feedback signal
- Immediate feedback for an action
- Can be positive, negative, or zero
- Guides learning toward desired behavior

**Policy (π)**: Strategy for choosing actions
- π(a|s): Probability of taking action a in state s
- Can be deterministic or stochastic
- The "brain" of the agent

**Value Function**: Expected future reward
- V(s): Value of being in state s
- Q(s,a): Value of taking action a in state s
- Helps evaluate how good states/actions are

### The Learning Objective

**Goal**: Find the optimal policy π* that maximizes expected cumulative reward

```
π* = argmax_π E[∑_{t=0}^∞ γ^t R_t]
```

Where:
- γ (gamma): Discount factor (0 ≤ γ ≤ 1)
- R_t: Reward at time t
- E[·]: Expected value

---

## The RL Learning Loop

### The Basic Learning Cycle

1. **Observe**: Agent observes current state
2. **Decide**: Agent selects action based on policy
3. **Act**: Agent executes action in environment
4. **Feedback**: Environment provides reward and new state
5. **Learn**: Agent updates its policy/value estimates
6. **Repeat**: Process continues

### Learning Mechanisms

**Model-Based Learning:**
- Learn a model of the environment
- Use model to plan and make decisions
- Examples: Dynamic Programming, Model Predictive Control

**Model-Free Learning:**
- Learn directly from experience
- No explicit model of environment
- Examples: Q-Learning, Policy Gradient methods

---

## Types of RL Problems

### Classification by Environment

**Fully Observable vs Partially Observable:**
- **Fully Observable**: Agent can see complete state
- **Partially Observable**: Agent has limited information

**Deterministic vs Stochastic:**
- **Deterministic**: Same action always produces same result
- **Stochastic**: Actions have probabilistic outcomes

**Episodic vs Continuing:**
- **Episodic**: Tasks have clear start and end
- **Continuing**: Tasks continue indefinitely

### Classification by Learning Approach

**Value-Based Methods:**
- Learn value functions (V or Q)
- Derive policy from values
- Examples: Q-Learning, SARSA

**Policy-Based Methods:**
- Learn policy directly
- Optimize policy parameters
- Examples: REINFORCE, Actor-Critic

**Model-Based Methods:**
- Learn environment model
- Use model for planning
- Examples: Dyna-Q, MCTS

---

## Mathematical Foundations

### Markov Property

A process has the Markov property if:
```
P(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, ...) = P(S_{t+1} | S_t, A_t)
```

**Implication**: Future depends only on current state and action, not history.

### Markov Decision Process (MDP)

An MDP is defined by:
- **States (S)**: Set of possible states
- **Actions (A)**: Set of possible actions
- **Transition Probabilities (P)**: P(s'|s,a)
- **Rewards (R)**: R(s,a,s') or R(s,a)
- **Discount Factor (γ)**: 0 ≤ γ ≤ 1

### Bellman Equations

**State Value Function:**
```
V^π(s) = E_π[∑_{k=0}^∞ γ^k R_{t+k+1} | S_t = s]
```

**Action Value Function:**
```
Q^π(s,a) = E_π[∑_{k=0}^∞ γ^k R_{t+k+1} | S_t = s, A_t = a]
```

**Bellman Equation for V^π:**
```
V^π(s) = ∑_a π(a|s) ∑_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]
```

**Bellman Equation for Q^π:**
```
Q^π(s,a) = ∑_{s'} P(s'|s,a)[R(s,a,s') + γ ∑_{a'} π(a'|s')Q^π(s',a')]
```

---

## Learning Objectives

### By the End of This Module, You Should Understand:

1. **Sequential Decision Making**
   - How agents make decisions over time
   - The role of consequences and uncertainty
   - Temporal nature of RL problems

2. **Multi-Armed Bandit Problem**
   - Simplest form of sequential decision making
   - Exploration vs exploitation trade-off
   - Action-value estimation

3. **Incremental Learning**
   - How to update estimates with new information
   - Step-size parameters and their effects
   - Convergence properties

4. **Exploration Strategies**
   - ε-greedy, UCB, and Thompson sampling
   - When to use each approach
   - Balancing exploration and exploitation

5. **Practical Implementation**
   - How to code basic RL algorithms
   - Performance evaluation metrics
   - Real-world applications

### Next Steps

This module provides the foundation for understanding sequential decision making. The next modules will cover:
- Markov Decision Processes (MDPs)
- Value Functions and Policies
- Dynamic Programming
- Monte Carlo Methods
- Temporal Difference Learning
- Deep Reinforcement Learning

### Exercises and Practice

1. **Implement ε-greedy bandit algorithm**
2. **Compare different exploration strategies**
3. **Experiment with step-size parameters**
4. **Apply to real-world problems**
5. **Analyze convergence behavior**

---

*This comprehensive guide covers the fundamentals of reinforcement learning, with a focus on sequential decision making as the foundation for more advanced topics.*
