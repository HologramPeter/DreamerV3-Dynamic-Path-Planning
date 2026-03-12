# DreamerV3 Navigation for Mobile Robots

## Overview

Research project investigating the use of **DreamerV3** as a probabilistic world model for adaptive path planning in autonomous mobile robots operating in dynamic environments.

The system integrates DreamerV3 with an ensemble of pre-trained navigation policies to enable real-time decision making under uncertainty. By performing imagined rollouts inside the world model, the framework evaluates multiple candidate policies and selects actions that remain effective under environmental changes such as moving obstacles.

To improve robustness against unreliable or missing sensor data, the project introduces a latent state recovery method using a fully connected neural network (FCN). The network reconstructs stable latent representations to maintain consistent navigation performance even when sensor inputs degrade.

Experiments were conducted using a simulated TurtleBot3 in the Webots environment. The ensemble-based approach was evaluated against individual base policies across metrics such as success rate, path efficiency, and trajectory smoothness. Results show that the DreamerV3-based ensemble achieves higher success rates and fewer steps to completion as environmental complexity increases.

## Tech Stack

- Python
- PyTorch
- Gymnasium
- NumPy
- Webots Simulator
