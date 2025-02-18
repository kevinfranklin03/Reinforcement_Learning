# 🧠 Implementation of Q-Learning in Grid World Navigation
*A Reinforcement Learning Approach Using Python*

## 📖 Overview
This project implements **Q-learning** for **grid world navigation**, a fundamental problem in **Reinforcement Learning (RL)**. The agent learns to navigate a grid world environment while maximizing cumulative rewards. 

The project explores **policy formulation, environment representation, observation definition, and action specification** while demonstrating **exploration vs. exploitation** strategies.

---

![image](https://github.com/user-attachments/assets/2127b949-6442-4707-b836-8bfe6c8e9af8)


## 📜 Abstract
Reinforcement Learning (RL) is an exciting field in AI where agents learn optimal decision-making in complex environments. This project presents a **Q-learning agent** solving a navigation task in a **grid world environment**. The agent iteratively updates its Q-values and learns to find the **optimal path** to maximize rewards.

Key topics covered:
- **Q-learning Algorithm**
- **Exploration vs. Exploitation**
- **Grid World Representation**
- **Epsilon-Greedy Exploration**
- **Reward Optimization**
- **Markov Decision Processes (MDP)**

---

## 🚀 Features
✅ **Q-learning implementation** for grid world navigation  
✅ **Custom reward structure** to balance learning  
✅ **Epsilon-greedy strategy** for controlled exploration  
✅ **Performance visualization** using reward graphs  
✅ Supports **obstacles, jump states, and terminal rewards**  

---

## ⚙️ Installation
Ensure you have Python **3.7+** installed.

1️⃣ **Clone the repository**
```bash
git clone https://github.com/your-username/Q-Learning-GridWorld.git

cd Q-Learning-GridWorld

python -m venv env

source env/bin/activate  # For Mac/Linux

env\Scripts\activate  # For Windows

pip install -r requirements.txt
```
1️⃣ Run the training script
```bash
python train.py
```

This script trains the Q-learning agent to navigate the grid world.

2️⃣ Visualize the learning process
```bash
python plot_results.py
```
This will generate a graph showing average reward per episode.

## 🎯 Grid World Environment
- **5x5 grid** with **obstacles**, **jump states**, and **goal states**.
- **Agent learns to find the optimal path** from start to goal.

### 🏆 Reward Structure:
| Event            | Reward  |
|-----------------|---------|
| ✅ **Goal State**   | +10    |
| ✅ **Jump Bonus**   | +5     |
| ❌ **Obstacle Penalty** | -1  |
| ➖ **Default Step Cost** | -1  |

---

## 📈 Training Process
- Uses the **epsilon-greedy exploration strategy**.
- Balances **exploration** (trying new paths) and **exploitation** (choosing the best-known path).
- Updates the **Q-table** iteratively for **optimal navigation**.

---

## 📌 Example Q-Table after training:
| State  | Action (Up) | Action (Down) | Action (Left) | Action (Right) |
|--------|------------|--------------|--------------|---------------|
| (2,1)  | 0.12       | 0.25         | -0.5         | 1.02          |
| (3,2)  | 0.45       | 0.12         | 0.8          | 0.96          |
| (5,5)  | ✅ Goal    | ✅ Goal      | ✅ Goal      | ✅ Goal       |

---

### 📌 Next Steps:
- **Copy and paste** this section into your `README.md` file.
- Update it if needed based on your latest Q-learning results.

🚀 *Happy Coding & Reinforcement Learning!* 🚀


## 📌 Algorithm
Q-learning is an **off-policy temporal difference (TD) reinforcement learning algorithm**.

### 🔣 Update Rule:
\[
Q(s, a) = Q(s, a) + \alpha \left[ R + \gamma \max Q(s', a') - Q(s, a) \right]
\]

Where:
- **\(Q(s, a)\)** → Q-value of the current state-action pair.
- **\(R\)** → Reward received after taking action **\(a\)**.
- **\(γ\)** → Discount factor (importance of future rewards).
- **\(α\)** → Learning rate (speed of learning).

---

## 🔍 Results & Insights
- **Training Convergence:** The **average reward increases** as the agent learns optimal paths.
- **Exploration-Exploitation Balance:** The **epsilon-greedy strategy** ensures the agent finds **better paths over time**.
- **Efficient Pathfinding:** The agent successfully **utilizes jump states** and avoids obstacles.

---

🔹 **Key Observations:**
- Early episodes → **More exploration**, lower rewards.
- Later episodes → **Better exploitation**, higher rewards.
- Final trained model **reaches the goal efficiently**.

---

## 🔮 Future Improvements
🚀 **Extend to larger grid sizes** (e.g., 10x10, 20x20).  
🤖 **Implement Deep Q-Networks (DQN)** for better decision-making.  
🎭 **Add multi-agent reinforcement learning** scenarios.  
📡 **Integrate real-world robotics simulations.**  



