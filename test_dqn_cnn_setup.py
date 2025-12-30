"""
Test script to verify DQN-CNN setup before full training.
"""

import sys
import os

# Ensure we can import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

print("=" * 60)
print("Testing DQN-CNN Setup")
print("=" * 60)

# -------------------------------------------------
# Test 1: Import environment
# -------------------------------------------------
print("\n1. Testing environment import...")
try:
    from src.environment.snake_env import SnakeEnvironment
    print("   ✓ SnakeEnvironment imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import SnakeEnvironment: {e}")
    sys.exit(1)

# -------------------------------------------------
# Test 2: Create environment
# -------------------------------------------------
print("\n2. Testing environment creation...")
try:
    env = SnakeEnvironment(grid_size=30, render=False, seed=42)
    print("   ✓ Environment created successfully")
    print(f"   - Grid size: {env.grid_size}x{env.grid_size}")
    print(f"   - Num actions: {env.get_num_actions()}")
except Exception as e:
    print(f"   ✗ Failed to create environment: {e}")
    sys.exit(1)

# -------------------------------------------------
# Test 3: Test CNN grid state
# -------------------------------------------------
print("\n3. Testing CNN grid state representation...")
try:
    env.reset()
    grid_state = env.get_grid_state()

    print("   ✓ Grid state retrieved")
    print(f"   - State shape: {grid_state.shape}")
    print(f"   - State dtype: {grid_state.dtype}")
    print(f"   - Unique values: {np.unique(grid_state)}")

    assert grid_state.shape == (3, env.grid_size, env.grid_size)
    print("   ✓ Grid state shape correct (3, H, W)")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# -------------------------------------------------
# Test 4: Test environment step with grid state
# -------------------------------------------------
print("\n4. Testing environment step...")
try:
    env.reset()
    state = env.get_grid_state()
    action = np.random.randint(0, 4)

    _, reward, done = env.step(action)
    next_state = env.get_grid_state()

    print(f"   ✓ Step executed | reward={reward:+.1f}, done={done}")
    print(f"   - Next state shape: {next_state.shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# -------------------------------------------------
# Test 5: Import CNN agent
# -------------------------------------------------
print("\n5. Testing CNN agent import...")
try:
    from src.agents.dqn_cnn import DQNCNNAgent
    print("   ✓ DQNCNNAgent imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import DQNCNNAgent: {e}")
    sys.exit(1)

# -------------------------------------------------
# Test 6: Create CNN agent
# -------------------------------------------------
print("\n6. Testing CNN agent creation...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DQNCNNAgent(
        grid_size=env.grid_size,
        action_size=4,
        device=device,
        seed=42
    )
    print("   ✓ CNN agent created successfully")
    print(f"   - Device: {device}")
    print(f"   - Epsilon: {agent.epsilon:.3f}")
    print(f"   - Replay buffer size: {agent.memory.buffer.maxlen}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# -------------------------------------------------
# Test 7: Forward pass through CNN
# -------------------------------------------------
print("\n7. Testing CNN forward pass...")
try:
    env.reset()
    state = env.get_grid_state()
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

    q_values = agent.policy_net(state_tensor)

    print("   ✓ Forward pass successful")
    print(f"   - Q-values shape: {q_values.shape}")
    assert q_values.shape == (1, 4)
    print("   ✓ Output shape correct (1, 4)")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# -------------------------------------------------
# Test 8: Replay buffer with CNN states
# -------------------------------------------------
print("\n8. Testing replay buffer with CNN states...")
try:
    for _ in range(20):
        agent.store_transition(
            state=np.random.rand(3, 30, 30).astype(np.float32),
            action=np.random.randint(0, 4),
            reward=float(np.random.randn()),
            next_state=np.random.rand(3, 30, 30).astype(np.float32),
            done=False
        )

    print(f"   ✓ Replay buffer populated ({len(agent.memory)} transitions)")
    states, actions, rewards, next_states, dones = agent.memory.sample(5)
    print(f"   ✓ Sampled batch shape: {states.shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# -------------------------------------------------
# Test 9: Action selection
# -------------------------------------------------
print("\n9. Testing action selection...")
try:
    env.reset()
    state = env.get_grid_state()

    action_train = agent.select_action(state, eval_mode=False)
    action_eval = agent.select_action(state, eval_mode=True)

    print(f"   ✓ Action selected (train): {action_train}")
    print(f"   ✓ Action selected (eval):  {action_eval}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# -------------------------------------------------
# Test 10: Mini CNN training loop
# -------------------------------------------------
print("\n10. Running mini CNN training loop (20 episodes)...")
try:
    env = SnakeEnvironment(grid_size=20, render=False, seed=42)
    agent = DQNCNNAgent(
        grid_size=env.grid_size,
        action_size=4,
        device=device,
        seed=42
    )

    scores = []

    for episode in range(20):
        env.reset()
        state = env.get_grid_state()
        done = False
        steps = 0

        while not done and steps < 100:
            action = agent.select_action(state)
            _, reward, done = env.step(action)
            next_state = env.get_grid_state()

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            steps += 1

        agent.update_epsilon()
        scores.append(env.get_score())

        if (episode + 1) % 5 == 0:
            print(
                f"   Episode {episode+1:2d}: "
                f"score={env.get_score()}, "
                f"steps={steps:3d}, "
                f"eps={agent.epsilon:.2f}"
            )

    print("   ✓ Mini CNN training completed")
    print(f"   - Avg score: {np.mean(scores):.2f}")
    print(f"   - Max score: {max(scores)}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL CNN TESTS PASSED!")
print("=" * 60)
print("\nCNN setup verified. Ready for full training!")
