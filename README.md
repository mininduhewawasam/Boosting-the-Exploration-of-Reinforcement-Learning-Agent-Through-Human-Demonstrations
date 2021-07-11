# Boosting the Exploration of Reinforcement Learning Agent Through Human Demonstrations

## Abstract

Exploration is a challenging task in reinforcement learning environments with sparse reward signals. This becomes exponential in long horizon tasks with complex state and action spaces which make the applicability of reinforcement learning in real world problems out of practical use. In this research work, we introduce a novel learning from demonstration approach to address this problem by combining human expertise and speed up the training process by minimizing the ignorant learning process of the reinforcement learning agent. The proposed method is built upon the Deep Deterministic Policy Gradient(DDPG) algorithm and Hindsight Experience replay(HER). This novel approach is successfully proven to outperform the learning efficiency of the baseline algorithm and accounts for the suboptimality of expert demonstration and minimizes overfitting bias of current approaches.

## Setting Up the Project

Clone the repository:

```setup
git clone https://github.com/mininduhewawasam/Boosting-the-Exploration-of-Reinforcement-Learning-Agent-Through-Human-Demonstrations.git
```

To install requirements:

```setup
pip install -r requirements.txt
```
Run this in the root dir to launch the application.:

```setup
mpirun -np $(nproc) python3 -u main.py
```

# Demo

Please find the demonstration of the prototype here
[Prototype demonstration](https://youtu.be/YTKjYaD3ntw)

# Presentaion

Please find the presentation of the research here
[Presentation](https://www.youtube.com/watch?v=ri-LCkBrwQA)

# Results

Please find the research results in the following link. 
[Results of the research](https://www.researchgate.net/publication/353167444_Boosting_The_Exploration_Of_Reinforcement_Learning_Agent_Through_Human_Demonstrations_A_dissertation_by)
