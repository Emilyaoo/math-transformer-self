# Math RL Transformer: 3x1 Multiplication Solver

This is my personal implementation of a reinforcement learning (RL) system using a Transformer-based model to solve 3-digit × 1-digit multiplication problems step by step.

---

## What This Project Does

- Trains an RL agent to reason through multi-step multiplication
- Solves problems like `123 × 7` using intermediate symbolic steps
- Uses a Transformer architecture as the policy network
- Includes a custom math environment and a GUI to display the solving process

---

## What I Changed or Added

- Rewrote the main.py to support 3-digit × 1-digit multiplication problems
- Integrated a Transformer-based policy network instead of a basic feedforward model
- Built a tkinter-based GUI to display the problem and the solving steps
- Cleaned up the previous project structure and kept only what was necessary
- Tuned the reward shaping to better encourage step-by-step reasoning
- Adjusted the tokenization logic to match multi-digit input handling
