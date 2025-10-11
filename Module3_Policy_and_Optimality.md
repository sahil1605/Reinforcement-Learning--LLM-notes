# Module 3: Reward, Policy, and Optimal Policy in MDPs

## Table of Contents
1. [Introduction to Policies](#introduction-to-policies)
2. [Policy in RL](#policy-in-rl)
3. [Optimal Policy](#optimal-policy)
4. [Value Functions & Optimality](#value-functions--optimality)
5. [Bellman Optimality Equations](#bellman-optimality-equations)
6. [Dynamic Programming for Optimal Policies](#dynamic-programming-for-optimal-policies)
7. [Generalized Policy Iteration (GPI)](#generalized-policy-iteration-gpi)
8. [Methods for Finding Optimal Policy](#methods-for-finding-optimal-policy)
9. [Intuitive Connections](#intuitive-connections)
10. [Practical Implementation](#practical-implementation)
11. [Summary and Key Takeaways](#summary-and-key-takeaways)

---

## Introduction to Policies

### The Ultimate Goal

In a Markov Decision Process (MDP), the ultimate goal is to find an **optimal policy** — a mapping from states to actions that maximizes the agent's expected return.

**Key Questions:**
- What is a policy?
- How do we define optimality?
- How do we find optimal policies?
- What methods are available?

### The Policy-Value Relationship

**Fundamental Insight**: A policy determines behavior, and behavior determines value. The optimal policy is the one that yields the highest value.

```
Policy π → Behavior → Value V^π(s)
Optimal Policy π* → Best Behavior → Optimal Value V*(s)
```

---

## Policy in RL

### Definition of Policy

A **policy π** defines the agent's behavior:

```
π(a|s) = P(A_t = a | S_t = s)
```

**Interpretation**: The probability of taking action a when in state s.

### Types of Policies

#### 1. Deterministic Policy

**Definition**: Always picks the same action in a given state.

```
π(s) = a
```

**Characteristics**:
- Single action per state
- No randomness in decision making
- Easier to implement and understand
- May not be optimal in all cases

**Example**: In a grid world, always move right when in state (0,0).

#### 2. Stochastic Policy

**Definition**: Assigns probabilities over actions in each state.

```
π(a|s) ∈ [0,1] and Σ_a π(a|s) = 1
```

**Characteristics**:
- Multiple actions possible per state
- Probabilistic decision making
- Can represent exploration
- More flexible than deterministic

**Example**: In a grid world, move right with 70% probability, up with 30% probability.

### Policy Representation

#### Tabular Policies

For finite state-action spaces:

```
π = {
    s1: {a1: 0.7, a2: 0.3},
    s2: {a1: 0.0, a2: 1.0},
    ...
}
```

#### Function-Based Policies

For continuous or large state spaces:

```
π(a|s) = f_θ(s, a)
```

Where f_θ is a parameterized function (e.g., neural network).

### Policy Properties

#### Valid Policy Requirements

1. **Probability Distribution**: Σ_a π(a|s) = 1 for all s
2. **Non-negative**: π(a|s) ≥ 0 for all s, a
3. **Defined for All States**: π(a|s) is defined for all reachable states

#### Policy Comparison

**Policy Ordering**: π₁ ≥ π₂ if and only if V^π₁(s) ≥ V^π₂(s) for all s

**Implication**: We can compare policies by comparing their value functions.

---

## Optimal Policy

### Definition of Optimal Policy

An **optimal policy π*** yields the highest expected return from every state.

**Mathematical Definition**:
```
π* = argmax_π V^π(s) for all s
```

### Properties of Optimal Policies

#### 1. Existence

**Theorem**: For any finite MDP, there exists at least one optimal policy.

**Proof**: The set of policies is finite, and the value function is bounded.

#### 2. Uniqueness of Value Function

**Theorem**: All optimal policies yield the same optimal value function V*(s).

**Implication**: While there may be multiple optimal policies, they all achieve the same performance.

#### 3. Deterministic Optimal Policies

**Theorem**: For any MDP, there exists a deterministic optimal policy.

**Implication**: We can focus on deterministic policies when searching for optimality.

### Multiple Optimal Policies

#### When Multiple Optimal Policies Exist

1. **Symmetric Actions**: When multiple actions yield the same expected return
2. **Equivalent Paths**: When different sequences of actions lead to the same outcome
3. **Redundant States**: When some states are equivalent in terms of value

#### Example: Grid World with Multiple Optimal Paths

```
Goal: [G]
Start: [S]
Obstacles: [X]

[ ][ ][G]
[ ][X][ ]
[S][ ][ ]
```

Multiple optimal policies exist:
- Policy 1: Right → Up → Right
- Policy 2: Up → Right → Up
- Both achieve the same optimal value

### Finding Optimal Policies

#### Brute Force Approach

**Method**: Enumerate all possible policies and compare their values.

**Limitations**:
- Exponential in number of states
- Computationally intractable for large MDPs
- Only feasible for very small problems

#### Dynamic Programming Approach

**Method**: Use Bellman equations to systematically find optimal policies.

**Advantages**:
- Polynomial time complexity
- Guaranteed to find optimal policy
- Efficient for medium-sized MDPs

---

## Value Functions & Optimality

### State-Value Function

**Definition**: Expected return starting from state s and following policy π:

```
V^π(s) = E_π[G_t | S_t = s]
```

**Interpretation**: How good is it to be in state s under policy π?

**Properties**:
- V^π(s) ≥ 0 if all rewards are non-negative
- V^π(s) represents the "value" of being in state s
- Helps compare different states

### Action-Value Function

**Definition**: Expected return starting from state s, taking action a, then following policy π:

```
Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
```

**Interpretation**: How good is it to take action a in state s under policy π?

**Properties**:
- Q^π(s,a) ≥ 0 if all rewards are non-negative
- Q^π(s,a) represents the "value" of action a in state s
- Helps compare different actions

### Optimal Value Functions

#### Optimal State-Value Function

```
V*(s) = max_π V^π(s)
```

**Interpretation**: The maximum value achievable from state s.

#### Optimal Action-Value Function

```
Q*(s,a) = max_π Q^π(s,a)
```

**Interpretation**: The maximum value achievable by taking action a in state s.

### Relationship Between V and Q

#### From Q to V

```
V^π(s) = Σ_a π(a|s) Q^π(s,a)
```

**Interpretation**: State value is the weighted average of action values.

#### From V to Q

```
Q^π(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]
```

**Interpretation**: Action value is the expected immediate reward plus discounted state value.

#### Optimal Relationship

```
V*(s) = max_a Q*(s,a)
```

**Interpretation**: Optimal state value equals the maximum optimal action value.

---

## Bellman Optimality Equations

### The Key Relationships

The Bellman Optimality Equations are the fundamental relationships that define optimal value functions.

#### Bellman Optimality Equation for V*

```
V*(s) = max_a Σ_{s',r} P(s',r|s,a)[r + γV*(s')]
```

**Interpretation**: The optimal value of state s is the maximum expected immediate reward plus discounted optimal value of the next state.

#### Bellman Optimality Equation for Q*

```
Q*(s,a) = Σ_{s',r} P(s',r|s,a)[r + γ max_{a'} Q*(s',a')]
```

**Interpretation**: The optimal value of action a in state s is the expected immediate reward plus discounted maximum optimal action value in the next state.

### Understanding the Equations

#### Components

1. **max_a**: Choose the best action
2. **Σ_{s',r} P(s',r|s,a)**: Expected value over next states and rewards
3. **r**: Immediate reward
4. **γV*(s')**: Discounted optimal value of next state

#### Recursive Nature

The equations are **recursive** because:
- V*(s) depends on V*(s') for all possible next states s'
- Q*(s,a) depends on Q*(s',a') for all possible next state-action pairs

#### Solution Methods

1. **Value Iteration**: Iteratively update V*(s) until convergence
2. **Policy Iteration**: Alternate between policy evaluation and improvement
3. **Linear Programming**: Solve as a linear program

### Bellman Equations for General Policies

#### Bellman Equation for V^π

```
V^π(s) = Σ_a π(a|s) Σ_{s',r} P(s',r|s,a)[r + γV^π(s')]
```

#### Bellman Equation for Q^π

```
Q^π(s,a) = Σ_{s',r} P(s',r|s,a)[r + γ Σ_{a'} π(a'|s')Q^π(s',a')]
```

**Key Difference**: General policies use weighted averages, optimal policies use maximums.

---

## Dynamic Programming for Optimal Policies

### Overview

Dynamic Programming (DP) methods use Bellman equations to compute or approximate optimal policies.

**Key Assumptions**:
- Complete model of the environment
- Finite state and action spaces
- Known transition probabilities and rewards

### Policy Evaluation

#### Goal

Estimate V^π(s) for a fixed policy π.

#### Algorithm

**Iterative Update**:
```
V_{k+1}(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a)[r + γV_k(s')]
```

**Steps**:
1. Initialize V_0(s) = 0 for all s
2. Iteratively update using the equation above
3. Continue until convergence (|V_{k+1}(s) - V_k(s)| < ε)

#### Convergence

**Theorem**: Policy evaluation converges to V^π(s) as k → ∞.

**Rate**: Geometric convergence with rate γ.

### Policy Improvement

#### Goal

Given V^π, find a better policy π'.

#### Algorithm

**Greedy Policy Improvement**:
```
π'(s) = argmax_a Σ_{s',r} P(s',r|s,a)[r + γV^π(s')]
```

**Steps**:
1. For each state s, find the action that maximizes expected return
2. Create new policy π' that is greedy with respect to V^π
3. The new policy is guaranteed to be at least as good as π

#### Policy Improvement Theorem

**Theorem**: If π' is greedy with respect to V^π, then V^π'(s) ≥ V^π(s) for all s.

**Proof**: By construction, π' chooses actions that maximize expected return.

### Policy Iteration

#### Algorithm

1. **Initialize**: Start with arbitrary policy π_0
2. **Policy Evaluation**: Compute V^π_k
3. **Policy Improvement**: Update policy to π_{k+1} = greedy(V^π_k)
4. **Repeat**: Until π_{k+1} = π_k (convergence)

#### Convergence

**Theorem**: Policy iteration converges to optimal policy π* in finite steps.

**Proof**: Each iteration improves the policy, and there are finitely many policies.

#### Advantages

- **Guaranteed Convergence**: Always finds optimal policy
- **Finite Steps**: Converges in at most |A|^|S| steps
- **Monotonic Improvement**: Each iteration improves the policy

#### Disadvantages

- **Computational Cost**: Each evaluation step may require many iterations
- **Memory Requirements**: Need to store full value function

### Value Iteration

#### Algorithm

**Combined Update**:
```
V_{k+1}(s) = max_a Σ_{s',r} P(s',r|s,a)[r + γV_k(s')]
```

**Steps**:
1. Initialize V_0(s) = 0 for all s
2. Iteratively update using the equation above
3. Continue until convergence (|V_{k+1}(s) - V_k(s)| < ε)
4. Extract optimal policy: π*(s) = argmax_a Q*(s,a)

#### Convergence

**Theorem**: Value iteration converges to V*(s) as k → ∞.

**Rate**: Geometric convergence with rate γ.

#### Advantages

- **Efficiency**: Combines evaluation and improvement
- **Simplicity**: Single update rule
- **Memory Efficient**: Only need to store current value function

#### Disadvantages

- **Slower Convergence**: May require more iterations than policy iteration
- **No Intermediate Policies**: Don't get policies until convergence

### Comparison of Methods

| Method | Convergence | Efficiency | Memory | Intermediate Results |
|--------|-------------|------------|---------|---------------------|
| **Policy Iteration** | Fast | High | High | Yes (policies) |
| **Value Iteration** | Slower | Medium | Low | No |
| **Brute Force** | N/A | Low | Low | Yes (all policies) |

---

## Generalized Policy Iteration (GPI)

### Concept

In practice, we don't fully separate policy evaluation and improvement. **Generalized Policy Iteration (GPI)** means policy evaluation and policy improvement happen together, iteratively.

### GPI Process

1. **Partial Policy Evaluation**: Update value function a few times
2. **Partial Policy Improvement**: Update policy based on current value function
3. **Repeat**: Continue alternating between evaluation and improvement
4. **Convergence**: Over time, the policy converges to optimal

### GPI Advantages

- **Flexibility**: Can adjust the balance between evaluation and improvement
- **Efficiency**: Don't need to fully evaluate before improving
- **Practical**: More realistic for real-world applications

### GPI Examples

#### Example 1: Alternating Updates

```
Step 1: Update V^π (partial evaluation)
Step 2: Update π (partial improvement)
Step 3: Update V^π (partial evaluation)
Step 4: Update π (partial improvement)
...
```

#### Example 2: Asynchronous Updates

```
Update V^π for some states
Update π for some states
Update V^π for other states
Update π for other states
...
```

---

## Methods for Finding Optimal Policy

### 1. Brute Force Search

**Method**: Compare all deterministic policies.

**Algorithm**:
1. Enumerate all possible deterministic policies
2. Evaluate each policy (compute V^π)
3. Select the policy with highest value

**Complexity**: O(|A|^|S|)

**Feasibility**: Only for very small MDPs (|S| < 10, |A| < 5)

### 2. Dynamic Programming

**Methods**: Policy Iteration, Value Iteration

**Advantages**:
- Guaranteed to find optimal policy
- Polynomial time complexity
- Efficient for medium-sized MDPs

**Limitations**:
- Requires complete model
- Limited to finite state-action spaces
- May be slow for large MDPs

### 3. Monte Carlo Methods

**Method**: Learn from sampled episodes, estimate returns

**Advantages**:
- No model required
- Can handle large state spaces
- Works with function approximation

**Limitations**:
- Requires episodic tasks
- May be slow to converge
- High variance in estimates

### 4. Temporal Difference (TD) Learning

**Method**: Bootstrapping from existing estimates

**Advantages**:
- No model required
- Works for both episodic and continuing tasks
- Can handle large state spaces
- Efficient learning

**Limitations**:
- May not converge to optimal policy
- Requires careful tuning
- Can be unstable with function approximation

### 5. Approximation & Deep RL

**Method**: Function approximators, neural networks

**Advantages**:
- Can handle very large state spaces
- Can learn complex policies
- State-of-the-art performance

**Limitations**:
- No convergence guarantees
- Requires careful design
- Computationally expensive

### Method Selection Guide

| Problem Size | Model Available | Method |
|--------------|----------------|---------|
| Small | Yes | Dynamic Programming |
| Medium | Yes | Dynamic Programming |
| Large | Yes | Approximate DP |
| Small | No | Monte Carlo |
| Medium | No | TD Learning |
| Large | No | Deep RL |

---

## Intuitive Connections

### Policy ↔ Value Function

**Connection 1**: A good policy yields a good value function.

**Intuition**: If the policy makes good decisions, the expected return will be high.

**Connection 2**: Optimal value function defines the optimal policy.

**Intuition**: If we know the optimal values, we can choose the best actions.

### Bellman Equations

**Connection**: Describe recursive relationships.

**Intuition**: The value of a state depends on the values of future states, creating a recursive structure.

**Basis**: All RL algorithms are based on Bellman equations in some form.

### Dynamic Programming

**Connection**: Systematic way to compute optimal policies in small MDPs.

**Intuition**: Break down the problem into smaller subproblems and solve them systematically.

**Foundation**: Provides the theoretical foundation for understanding RL algorithms.

---

## Practical Implementation

### Policy Iteration Implementation

```python
import numpy as np
from typing import Dict, List, Tuple

class PolicyIteration:
    def __init__(self, mdp, gamma=0.9, epsilon=1e-6):
        self.mdp = mdp
        self.gamma = gamma
        self.epsilon = epsilon
        self.V = {s: 0.0 for s in mdp.states}
        self.policy = {s: mdp.actions[0] for s in mdp.states}  # Random initial policy
    
    def policy_evaluation(self):
        """Evaluate current policy"""
        while True:
            V_old = self.V.copy()
            
            for state in self.mdp.states:
                if not self.mdp.is_terminal(state):
                    value = 0
                    for action in self.mdp.get_actions(state):
                        action_prob = 1.0 if self.policy[state] == action else 0.0
                        for next_state, reward, prob in self.mdp.get_transitions(state, action):
                            value += action_prob * prob * (reward + self.gamma * V_old[next_state])
                    
                    self.V[state] = value
            
            # Check convergence
            max_change = max(abs(self.V[s] - V_old[s]) for s in self.mdp.states)
            if max_change < self.epsilon:
                break
    
    def policy_improvement(self):
        """Improve policy based on current value function"""
        policy_stable = True
        
        for state in self.mdp.states:
            if not self.mdp.is_terminal(state):
                old_action = self.policy[state]
                
                # Find best action
                best_action = None
                best_value = float('-inf')
                
                for action in self.mdp.get_actions(state):
                    action_value = 0
                    for next_state, reward, prob in self.mdp.get_transitions(state, action):
                        action_value += prob * (reward + self.gamma * self.V[next_state])
                    
                    if action_value > best_value:
                        best_value = action_value
                        best_action = action
                
                self.policy[state] = best_action
                
                if old_action != best_action:
                    policy_stable = False
        
        return policy_stable
    
    def solve(self):
        """Solve MDP using policy iteration"""
        iteration = 0
        
        while True:
            print(f"Iteration {iteration}: Policy Evaluation")
            self.policy_evaluation()
            
            print(f"Iteration {iteration}: Policy Improvement")
            policy_stable = self.policy_improvement()
            
            iteration += 1
            
            if policy_stable:
                print(f"Converged after {iteration} iterations")
                break
        
        return self.policy, self.V
```

### Value Iteration Implementation

```python
class ValueIteration:
    def __init__(self, mdp, gamma=0.9, epsilon=1e-6):
        self.mdp = mdp
        self.gamma = gamma
        self.epsilon = epsilon
        self.V = {s: 0.0 for s in mdp.states}
    
    def solve(self):
        """Solve MDP using value iteration"""
        iteration = 0
        
        while True:
            V_old = self.V.copy()
            
            for state in self.mdp.states:
                if not self.mdp.is_terminal(state):
                    max_value = float('-inf')
                    
                    for action in self.mdp.get_actions(state):
                        action_value = 0
                        for next_state, reward, prob in self.mdp.get_transitions(state, action):
                            action_value += prob * (reward + self.gamma * V_old[next_state])
                        
                        max_value = max(max_value, action_value)
                    
                    self.V[state] = max_value
            
            # Check convergence
            max_change = max(abs(self.V[s] - V_old[s]) for s in self.mdp.states)
            iteration += 1
            
            print(f"Iteration {iteration}: Max change = {max_change:.6f}")
            
            if max_change < self.epsilon:
                print(f"Converged after {iteration} iterations")
                break
        
        # Extract optimal policy
        policy = {}
        for state in self.mdp.states:
            if not self.mdp.is_terminal(state):
                best_action = None
                best_value = float('-inf')
                
                for action in self.mdp.get_actions(state):
                    action_value = 0
                    for next_state, reward, prob in self.mdp.get_transitions(state, action):
                        action_value += prob * (reward + self.gamma * self.V[next_state])
                    
                    if action_value > best_value:
                        best_value = action_value
                        best_action = action
                
                policy[state] = best_action
            else:
                policy[state] = None
        
        return policy, self.V
```

### Grid World Example

```python
def create_grid_world_mdp(width, height, goal, obstacles, gamma=0.9):
    """Create a grid world MDP for testing"""
    states = [(x, y) for x in range(width) for y in range(height)]
    actions = ['up', 'down', 'left', 'right']
    
    mdp = MDP(states, actions, gamma)
    
    # Add transitions and rewards
    for state in states:
        if state == goal:
            continue  # Terminal state
        
        for action in actions:
            next_state = get_next_state(state, action, width, height)
            
            if next_state in obstacles:
                # Hit obstacle, stay in place
                mdp.add_transition(state, action, state, -10, 1.0)
            elif next_state == goal:
                # Reach goal
                mdp.add_transition(state, action, next_state, 100, 1.0)
            else:
                # Normal movement
                mdp.add_transition(state, action, next_state, -1, 1.0)
    
    return mdp

# Example usage
grid_world = create_grid_world_mdp(4, 4, (3, 3), [(1, 1), (2, 1)])

# Solve using policy iteration
pi_solver = PolicyIteration(grid_world)
policy_pi, V_pi = pi_solver.solve()

# Solve using value iteration
vi_solver = ValueIteration(grid_world)
policy_vi, V_vi = vi_solver.solve()

# Compare results
print("Policy Iteration Policy:", policy_pi)
print("Value Iteration Policy:", policy_vi)
print("Policies are equal:", policy_pi == policy_vi)
```

---

## Summary and Key Takeaways

### Core Concepts

1. **Policy**: Defines agent behavior (deterministic or stochastic)
2. **Optimal Policy**: Maximizes expected return from every state
3. **Value Functions**: V^π(s) and Q^π(s,a) represent expected returns
4. **Bellman Equations**: Recursive relationships for value functions
5. **Dynamic Programming**: Systematic methods for finding optimal policies

### Key Relationships

- **Policy → Value**: Good policy yields good value function
- **Value → Policy**: Optimal value function defines optimal policy
- **Bellman Equations**: Foundation for all RL algorithms
- **Dynamic Programming**: Systematic approach to optimality

### Methods for Finding Optimal Policies

1. **Brute Force**: Enumerate all policies (only for tiny MDPs)
2. **Dynamic Programming**: Policy iteration, value iteration
3. **Monte Carlo**: Learn from experience (no model required)
4. **Temporal Difference**: Bootstrapping methods
5. **Deep RL**: Function approximation for large problems

### Practical Considerations

- **Model Requirements**: DP needs complete model, others don't
- **Computational Complexity**: Trade-offs between methods
- **Convergence Guarantees**: DP guarantees optimality, others may not
- **Scalability**: Different methods scale differently

### Next Steps

This module provides the foundation for understanding policies and optimality. The next modules will cover:
- **Monte Carlo Methods**: Learning from experience
- **Temporal Difference Learning**: Q-learning, SARSA
- **Function Approximation**: Handling large state spaces
- **Deep Reinforcement Learning**: Neural network approaches

### Exercises

1. **Implement Policy Iteration** for a simple grid world
2. **Compare Policy vs Value Iteration** in terms of convergence
3. **Analyze Multiple Optimal Policies** in symmetric environments
4. **Experiment with Different Discount Factors** and their effects
5. **Visualize Policy Convergence** over iterations

---

*This module establishes the theoretical foundation for understanding policies and optimality in MDPs, providing the mathematical tools and practical methods for finding optimal policies through dynamic programming.*
