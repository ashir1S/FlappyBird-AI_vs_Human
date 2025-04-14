# FlappyBird-AI_vs_Human

This project implements Deep Q-Learning (DQN) to train an AI agent to play **Flappy Bird**, along with a human-playable mode. The agent learns through trial and error, maximizing its rewards using the Q-learning algorithm. You can switch between human and AI modes to either play the game yourself or watch the AI agent play.

🔗 **GitHub Repository:** [FlappyBird-AI_vs_Human](https://github.com/ashir1S/FlappyBird-AI_vs_Human)

## Installation

### Dependencies

To run this project, ensure you have the following dependencies installed:

- Python 2.7 or 3.x
- TensorFlow 0.7 (or a compatible version)
- pygame
- OpenCV-Python

You can install the dependencies using pip:

```bash
pip install tensorflow==0.7 pygame opencv-python
```

### Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/ashir1S/FlappyBird-AI_vs_Human.git
   cd FlappyBird-AI_vs_Human
   ```

2. Install the necessary Python dependencies by running the command above.
3. Ensure you have all game assets (images, sounds, etc.) in the appropriate directories.

## Modes

This project supports two gameplay modes:

- **AI Mode:** Watch the trained agent play Flappy Bird using Deep Q-Learning.
- **Human Mode:** Play Flappy Bird manually using keyboard controls.

You can toggle between the two modes by modifying the `human_mode` flag in the main script.

## How DQN Works

The Deep Q-Learning agent learns by interacting with the **Flappy Bird** game environment. Here's how it works:

1. **Convolutional Neural Network (CNN):**  
   Processes raw game frames and predicts Q-values, representing the expected future reward for each action in a given state.

2. **ϵ-greedy Policy:**  
   Starts with high ϵ (mostly random actions) for exploration. Over time, ϵ decays, shifting towards exploiting learned strategies.

3. **Q-learning:**  
   Updates Q-values based on observed rewards and estimated future rewards.

4. **Experience Replay:**  
   Stores `(state, action, reward, next_state)` tuples in a replay memory and samples random mini-batches to break correlations between consecutive experiences.

5. **Training Algorithm:**  
   Uses **Gradient Descent** (Adam optimizer, learning rate = `0.000001`) to minimize the difference between predicted Q-values and target Q-values.

## Training Process

### Steps

1. **ϵ-greedy Exploration:**  
   Begin with ϵ = 1.0 and decay it over time toward a minimum value.

2. **Network Architecture:**  
   - 3 convolutional layers  
   - 1 fully connected layer (outputs Q-values for each action)

3. **Optimization:**  
   - Optimizer: Adam  
   - Learning rate: `0.000001`

4. **Experience Replay:**  
   Sample random batches from replay memory for training.

5. **Training Loop:**  
   Continuously play the game, collect experiences, and update network weights.

### Hyperparameters

- **ϵ (epsilon):** Starts at 1.0, decays to a lower bound.
- **Learning Rate:** `0.000001`
- **Batch Size:** Adjustable based on your hardware.
- **Replay Memory Size:** Adjustable (e.g., 50,000 frames).
- **Network Architecture:** 3 conv layers + 1 FC layer.

## FAQ

### How to Load Checkpoints

If you encounter a checkpoint error, update the `model_checkpoint_path` in `saved_networks/checkpoint` to:

```
saved_networks/bird-dqn-2920000
```

### Troubleshooting

- **Missing Dependencies:**  
  Verify you’ve installed all required libraries (see Installation).

- **Performance Issues:**  
  Try reducing batch size or use a GPU for faster training.

## References

- **Human-level Control through Deep Reinforcement Learning**, Mnih et al., *Nature*, 2015.
- **Playing Atari with Deep Reinforcement Learning**, Mnih et al., *NIPS*, 2013.