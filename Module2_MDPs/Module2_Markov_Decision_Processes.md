# Module 2: Markov Decision Processes (MDPs) — The Foundation of Reinforcement Learning

## Table of Contents
1. [Introduction to MDPs](#introduction-to-mdps)
2. [What is an MDP?](#what-is-an-mdp)
3. [The Agent-Environment Loop](#the-agent-environment-loop)
4. [The Goal: Maximizing Return](#the-goal-maximizing-return)
5. [Episodic vs Continuing Tasks](#episodic-vs-continuing-tasks)
6. [Value Functions](#value-functions)
7. [MDPs vs Bandits: Why MDPs Matter](#mdps-vs-bandits-why-mdps-matter)
8. [Mathematical Foundations](#mathematical-foundations)
9. [Practical Examples](#practical-examples)
10. [Implementation and Code](#implementation-and-code)
11. [Summary and Key Takeaways](#summary-and-key-takeaways)

---

## Introduction to MDPs

### From Bandits to Sequential Decision Making

The k-armed bandit problem gives us a way to learn action-values for static, independent actions. But in real-world problems, actions often change the state of the environment — and that's where Markov Decision Processes (MDPs) come in.

**Key Difference:**
- **Bandits**: Actions are independent, no state transitions
- **MDPs**: Actions affect future states, creating sequential dependencies

### Why MDPs Matter

MDPs provide the mathematical foundation for:
- **Sequential Decision Making**: Actions affect future states
- **Planning**: Reasoning about consequences of actions
- **Learning**: Understanding how to optimize long-term behavior
- **Real-world Applications**: Robotics, games, autonomous systems

---

## What is an MDP?

### Definition

A Markov Decision Process (MDP) is a mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision maker.

### MDP Components

An MDP is defined by the tuple: **⟨S, A, P, R, γ⟩**

#### 1. **S → Set of States**
- All possible situations the agent can be in
- Can be finite or infinite
- Examples: positions on a grid, game board configurations, robot poses

#### 2. **A → Set of Actions**
- All possible actions the agent can take
- Can be discrete or continuous
- Examples: move left/right, buy/sell, accelerate/brake

#### 3. **P(s'|s,a) → Transition Probabilities**
- Probability of moving to state s' given action a in state s
- P(s'|s,a) = P(S_{t+1} = s' | S_t = s, A_t = a)
- Must satisfy: Σ_{s'} P(s'|s,a) = 1 for all s, a

#### 4. **R(s,a) → Reward Function**
- Expected immediate reward for taking action a in state s
- Can be deterministic or stochastic
- Guides the agent toward desired behavior

#### 5. **γ → Discount Factor**
- How much we care about future rewards vs immediate ones
- 0 ≤ γ ≤ 1
- γ = 0: Only care about immediate reward
- γ = 1: Equal weight to all future rewards
- γ ≈ 0.9: Common choice for most problems

### The Markov Property

A process has the **Markov Property** if:
```
P(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, ...) = P(S_{t+1} | S_t, A_t)
```

**Intuition**: The future depends only on the current state and action, not the history.

**Implications**:
- Current state contains all relevant information
- No need to remember past states
- Enables efficient algorithms

---

## The Agent-Environment Loop

### The Basic Interaction Cycle

At each time step t:

1. **Agent observes current state S_t**
2. **Agent chooses an action A_t**
3. **Environment returns reward R_{t+1} and new state S_{t+1}**

This can be summarized as:
```
S_t → A_t → (R_{t+1}, S_{t+1})
```

### Visual Representation

```
┌─────────┐    Action A_t    ┌─────────────┐    Reward R_{t+1}    ┌─────────┐
│  Agent  │ ──────────────→  │ Environment │ ──────────────────→  │  Agent  │
└─────────┘                  └─────────────┘    State S_{t+1}      └─────────┘
     ↑                                                                    │
     │                                                                   │
     └─────────────────── Observe State S_t ────────────────────────────┘
```

### Detailed Step-by-Step Process

1. **Observation**: Agent receives state S_t
2. **Decision**: Agent selects action A_t based on policy π
3. **Execution**: Agent performs action A_t
4. **Feedback**: Environment provides:
   - Immediate reward R_{t+1}
   - New state S_{t+1}
5. **Learning**: Agent updates its knowledge
6. **Repeat**: Process continues

### Example: Grid World

```
Initial State: (0,0)
Action: Move Right
Result: New State (1,0), Reward +1
```

---

## The Goal: Maximizing Return

### What is Return?

The **return G_t** is the total reward the agent expects to receive from time t onwards:

**Undiscounted Return:**
```
G_t = R_{t+1} + R_{t+2} + R_{t+3} + ...
```

**Discounted Return:**
```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
```

### Understanding the Discount Factor γ

**γ = 0**: Only immediate reward matters
- G_t = R_{t+1}
- Myopic behavior
- Good for: One-step decisions

**γ = 1**: All future rewards matter equally
- G_t = R_{t+1} + R_{t+2} + R_{t+3} + ...
- Long-term thinking
- Good for: Episodic tasks

**γ ≈ 0.9**: Balanced approach
- Recent rewards matter more than distant ones
- Good for: Most practical problems

### The Agent's Objective

**Goal**: Find the policy π* that maximizes expected return:

```
π* = argmax_π E[G_t | S_t = s]
```

This means finding the strategy that gives the highest expected total reward.

---

## Episodic vs Continuing Tasks

### Episodic Tasks

**Definition**: Tasks that have a natural ending point (episode).

**Characteristics**:
- Finite sequence of steps
- Clear start and end
- Return is sum of rewards within episode
- Examples: Games, navigation to goal, completing a task

**Example**: Chess Game
- Episode starts: Game begins
- Episode ends: Checkmate, stalemate, or draw
- Return: +1 for win, -1 for loss, 0 for draw

**Mathematical Form**:
```
G_t = R_{t+1} + R_{t+2} + ... + R_T
```
Where T is the terminal time step.

### Continuing Tasks

**Definition**: Tasks that continue indefinitely without natural termination.

**Characteristics**:
- No natural end
- Agent acts forever
- Return is discounted infinite sum
- Examples: Stock trading, robot maintenance, resource management

**Example**: Stock Trading
- No natural end point
- Agent continuously makes buy/sell decisions
- Return: Discounted sum of all future profits

**Mathematical Form**:
```
G_t = Σ_{k=0}^∞ γ^k R_{t+k+1}
```

### Choosing Between Episodic and Continuing

**Use Episodic When**:
- Task has clear start/end
- Natural termination conditions
- Want to measure episode performance

**Use Continuing When**:
- Task runs indefinitely
- No natural stopping point
- Focus on long-term average performance

---

## Value Functions

### Why Value Functions?

Since we cannot predict rewards perfectly, we learn the **expected return** instead of trying to predict exact rewards.

### State-Value Function V^π(s)

**Definition**: Expected return starting from state s and following policy π:

```
V^π(s) = E_π[G_t | S_t = s]
```

**Intuition**: How good is it to be in state s?

**Properties**:
- V^π(s) ≥ 0 if all rewards are non-negative
- V^π(s) represents the "value" of being in state s
- Helps compare different states

### Action-Value Function Q^π(s,a)

**Definition**: Expected return starting from state s, taking action a, then following policy π:

```
Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
```

**Intuition**: How good is it to take action a in state s?

**Properties**:
- Q^π(s,a) ≥ 0 if all rewards are non-negative
- Q^π(s,a) represents the "value" of action a in state s
- Helps compare different actions

### Relationship Between V and Q

**From Q to V**:
```
V^π(s) = Σ_a π(a|s) Q^π(s,a)
```

**From V to Q**:
```
Q^π(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]
```

### Optimal Value Functions

**Optimal State-Value Function**:
```
V*(s) = max_π V^π(s)
```

**Optimal Action-Value Function**:
```
Q*(s,a) = max_π Q^π(s,a)
```

**Key Insight**: V*(s) = max_a Q*(s,a)

---

## MDPs vs Bandits: Why MDPs Matter

### Bandit Problems

**Characteristics**:
- Actions are independent
- No state transitions
- Each action has fixed reward distribution
- Good for: A/B testing, ad selection, simple optimization

**Example**: Website A/B Testing
- Action 1: Show version A
- Action 2: Show version B
- No state changes based on actions
- Each action has independent reward

### MDP Problems

**Characteristics**:
- Actions affect future states
- Sequential decision making
- Current action influences future options
- Good for: Robotics, games, navigation, control

**Example**: Robot Navigation
- Action: Move left
- Result: Robot position changes
- Future actions depend on new position
- Sequential consequences matter

### When to Use Each

**Use Bandits When**:
- Actions are independent
- No state transitions
- Simple optimization problems
- A/B testing scenarios

**Use MDPs When**:
- Actions affect future states
- Sequential decision making
- Planning and navigation
- Complex control problems

---

## Mathematical Foundations

### Bellman Equations

The Bellman equations are the fundamental equations that relate value functions to themselves.

#### Bellman Equation for V^π

```
V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]
```

**Interpretation**: The value of state s is the expected immediate reward plus the discounted value of the next state.

#### Bellman Equation for Q^π

```
Q^π(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ Σ_{a'} π(a'|s')Q^π(s',a')]
```

**Interpretation**: The value of action a in state s is the expected immediate reward plus the discounted value of the next state-action pair.

### Optimality Equations

#### Bellman Optimality Equation for V*

```
V*(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV*(s')]
```

#### Bellman Optimality Equation for Q*

```
Q*(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ max_{a'} Q*(s',a')]
```

### Policy Improvement

Given a policy π, we can find a better policy π' by:

```
π'(s) = argmax_a Q^π(s,a)
```

**Theorem**: If π' is greedy with respect to V^π, then V^π'(s) ≥ V^π(s) for all s.

---

## Practical Examples

### Example 1: Grid World Navigation

**Problem**: Agent must navigate from start to goal in a grid world.

**States**: Grid positions (x,y)
**Actions**: Move up, down, left, right
**Rewards**: +10 for reaching goal, -1 for each step
**Transitions**: Deterministic movement

**MDP Components**:
- S = {(0,0), (0,1), ..., (4,4)}
- A = {up, down, left, right}
- P(s'|s,a) = 1 if s' is the result of action a in state s
- R(s,a) = +10 if s is goal, -1 otherwise
- γ = 0.9

### Example 2: Inventory Management

**Problem**: Store manager must decide how much inventory to order.

**States**: Current inventory level
**Actions**: Order quantities (0, 10, 20, 30, 40)
**Rewards**: Profit from sales minus ordering costs
**Transitions**: Stochastic demand affects inventory

**MDP Components**:
- S = {0, 10, 20, 30, 40, 50} (inventory levels)
- A = {0, 10, 20, 30, 40} (order quantities)
- P(s'|s,a) depends on demand distribution
- R(s,a) = sales revenue - ordering cost
- γ = 0.95

### Example 3: Robot Control

**Problem**: Robot must navigate to target while avoiding obstacles.

**States**: Robot position and orientation
**Actions**: Move forward, turn left, turn right
**Rewards**: +100 for reaching target, -10 for hitting obstacle
**Transitions**: Stochastic movement with noise

**MDP Components**:
- S = {(x,y,θ) | x,y ∈ grid, θ ∈ {0°,90°,180°,270°}}
- A = {forward, left, right}
- P(s'|s,a) includes movement noise
- R(s,a) = +100 for target, -10 for obstacle, -1 for step
- γ = 0.9

---

## Implementation and Code

### Basic MDP Class

```python
import numpy as np
from typing import Dict, List, Tuple, Optional

class MDP:
    def __init__(self, states: List, actions: List, transitions: Dict, 
                 rewards: Dict, gamma: float = 0.9):
        self.states = states
        self.actions = actions
        self.transitions = transitions  # P(s'|s,a)
        self.rewards = rewards  # R(s,a,s')
        self.gamma = gamma
    
    def get_transition_prob(self, state, action, next_state):
        """Get P(s'|s,a)"""
        return self.transitions.get((state, action, next_state), 0.0)
    
    def get_reward(self, state, action, next_state):
        """Get R(s,a,s')"""
        return self.rewards.get((state, action, next_state), 0.0)
    
    def get_possible_actions(self, state):
        """Get actions available in state"""
        return [a for a in self.actions 
                if any(self.get_transition_prob(state, a, s) > 0 
                      for s in self.states)]
    
    def get_possible_next_states(self, state, action):
        """Get possible next states from (state, action)"""
        return [s for s in self.states 
                if self.get_transition_prob(state, action, s) > 0]
```

### Value Iteration Algorithm

```python
def value_iteration(mdp: MDP, epsilon: float = 1e-6, max_iterations: int = 1000):
    """
    Value Iteration algorithm to find optimal value function
    """
    V = {s: 0.0 for s in mdp.states}
    
    for iteration in range(max_iterations):
        V_old = V.copy()
        
        for state in mdp.states:
            if state in mdp.get_possible_actions(state):  # Not terminal
                Q_values = []
                for action in mdp.get_possible_actions(state):
                    q_value = 0
                    for next_state in mdp.get_possible_next_states(state, action):
                        prob = mdp.get_transition_prob(state, action, next_state)
                        reward = mdp.get_reward(state, action, next_state)
                        q_value += prob * (reward + mdp.gamma * V_old[next_state])
                    Q_values.append(q_value)
                
                V[state] = max(Q_values) if Q_values else 0
        
        # Check for convergence
        max_change = max(abs(V[s] - V_old[s]) for s in mdp.states)
        if max_change < epsilon:
            print(f"Converged after {iteration + 1} iterations")
            break
    
    return V
```

### Policy Extraction

```python
def extract_policy(mdp: MDP, V: Dict) -> Dict:
    """
    Extract optimal policy from value function
    """
    policy = {}
    
    for state in mdp.states:
        if state in mdp.get_possible_actions(state):  # Not terminal
            Q_values = {}
            for action in mdp.get_possible_actions(state):
                q_value = 0
                for next_state in mdp.get_possible_next_states(state, action):
                    prob = mdp.get_transition_prob(state, action, next_state)
                    reward = mdp.get_reward(state, action, next_state)
                    q_value += prob * (reward + mdp.gamma * V[next_state])
                Q_values[action] = q_value
            
            # Greedy policy: choose action with highest Q-value
            policy[state] = max(Q_values, key=Q_values.get)
        else:
            policy[state] = None  # Terminal state
    
    return policy
```

### Grid World Example

```python
def create_grid_world(width: int, height: int, goal: Tuple, obstacles: List):
    """
    Create a grid world MDP
    """
    states = [(x, y) for x in range(width) for y in range(height)]
    actions = ['up', 'down', 'left', 'right']
    
    transitions = {}
    rewards = {}
    
    for state in states:
        for action in actions:
            next_state = get_next_state(state, action, width, height)
            
            if next_state in obstacles:
                # Hit obstacle, stay in place
                transitions[(state, action, state)] = 1.0
                rewards[(state, action, state)] = -10
            elif next_state == goal:
                # Reach goal
                transitions[(state, action, next_state)] = 1.0
                rewards[(state, action, next_state)] = 100
            else:
                # Normal movement
                transitions[(state, action, next_state)] = 1.0
                rewards[(state, action, next_state)] = -1
    
    return MDP(states, actions, transitions, rewards, gamma=0.9)

def get_next_state(state, action, width, height):
    """Get next state after taking action"""
    x, y = state
    
    if action == 'up':
        return (x, min(y + 1, height - 1))
    elif action == 'down':
        return (x, max(y - 1, 0))
    elif action == 'left':
        return (max(x - 1, 0), y)
    elif action == 'right':
        return (min(x + 1, width - 1), y)
    
    return state

# Example usage
grid_world = create_grid_world(5, 5, (4, 4), [(2, 2), (3, 2)])
V = value_iteration(grid_world)
policy = extract_policy(grid_world, V)
```

---

## Summary and Key Takeaways

### Core Concepts

1. **MDPs Model Sequential Decision Making**: Actions affect future states
2. **Five Components**: States, Actions, Transitions, Rewards, Discount factor
3. **Value Functions**: V^π(s) and Q^π(s,a) represent expected returns
4. **Bellman Equations**: Fundamental relationships between value functions
5. **Optimality**: Finding policies that maximize expected return

### Key Differences from Bandits

| Aspect | Bandits | MDPs |
|--------|---------|------|
| **State Changes** | No | Yes |
| **Sequential Dependencies** | No | Yes |
| **Planning Horizon** | Immediate | Long-term |
| **Applications** | A/B testing, ads | Robotics, games, control |

### Mathematical Foundations

- **Markov Property**: Future depends only on current state
- **Bellman Equations**: Recursive relationships for value functions
- **Value Iteration**: Algorithm to find optimal values
- **Policy Extraction**: Derive optimal policy from values

### Practical Applications

1. **Robotics**: Navigation, manipulation, control
2. **Games**: Chess, Go, video games
3. **Finance**: Portfolio management, trading
4. **Operations**: Inventory management, scheduling
5. **Autonomous Systems**: Self-driving cars, drones

### Next Steps

This module provides the foundation for understanding MDPs. The next modules will cover:
- **Dynamic Programming**: Value iteration, policy iteration
- **Monte Carlo Methods**: Learning from experience
- **Temporal Difference Learning**: Q-learning, SARSA
- **Function Approximation**: Deep reinforcement learning

### Exercises

1. **Implement Value Iteration** for a simple grid world
2. **Compare Different Discount Factors** and their effects
3. **Design MDPs** for real-world problems
4. **Analyze Convergence** of value iteration algorithm
5. **Extract and Visualize Policies** from learned value functions

---

*This module establishes the mathematical foundation of reinforcement learning through Markov Decision Processes, providing the framework for understanding sequential decision making and value-based learning algorithms.*
