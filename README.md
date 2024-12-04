## Deep RL Bidding for Multi-Agent Coordination in Robotics

### Problem Statement

Recent advancements in single-agent vision-language-action models, such as π0 [LF24], have demonstrated promise in robotics towards more generalizable task completion. In conjunction with decreasing hardware costs, more templatized robotic forms like Stanford’s Pupper [Kau22] and wider robotic literacy, an existing challenge is demonstrating generalize-able strategic coordination of multiple-agents towards a set of high-level objectives [OH23].

A single illustrative example is the user-prompt ’clean the kitchen’, over a set of agents that might include autonomous carts with speed and large storage capacity, humanoids with hand-like manipulators and cloud-integrated appliances. Each robot will likely have independent knowledge of its capabilities and state, but not of other agents hence a requirement for decentralized control.

### Proposal Overview

Here we offer a novel framework for task distribution in multi-agent systems using bidding mechanisms. Borrowing on existing game theory literature, bidding mechanisms provide guarantees on utility and adversarial behavior dependent on market structure \cite{7}. In this work, we utilize an instantaneous first-price auction with $B$ bidders and $T$ tasks and consider constraints such as distance $D(B_i, T_j) = d_{ij}$ and type match $M(B_i, T_j) = I(B_i, T_j) = m_{ij}$. A more general problem statement might include additional binary or variable constraints: task time to complete, 'energy' consumption, internet access, etc. We also exclude considerations of sequential task completion, hence assume $B > T$ and time for each task is equal, $\text{time}(T) = t$.

<p align='center'>
  <img width="75%" alt="Task_Allocation" src="https://github.com/user-attachments/assets/16e5da56-b633-4e07-9538-cce0b1e6a697">
</p>

### Navigating Repo

To get started, feel free to enter notebook 'purchasing_v1.ipynb'. For each case I consider, I clearly label 3-4 cells that should clearly document my work.

More generally, this repo is structured with an envs and runs folder. Envs contains each custom environment I've built to engage with bidding, namely PurchasingEnv() and MultiAgent_PurchaseEnv(). You can import them by opening a notebook at the root directory and running:

```
from envs.purchasing import PurchaseEnv, MultiAgent_PurchaseEnv
```

Each environment is formatted as a normal [custom gym environment](https://www.gymlibrary.dev/content/environment_creation/) and you can engage with it via `env.step()` and `env.render()` after importing the env. 

Finally for model training, I suggest using [Stable Baselines&#39;s Guide for Custom Environments](https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html) which will help get you started.

### Contributors

Note this problem statement was initially collaborated on as a CalHacks 2024 project with Ojas Karnavat, Agam Gambhir and Abhinav Goel. Since then it has been developed and transformed such that none of the original code is being used, however the original code remains located at `./old_versions/v1/v1.ipynb`. As of Dec 2024, this project is primarily owned and maintained by Devin Gupta.
