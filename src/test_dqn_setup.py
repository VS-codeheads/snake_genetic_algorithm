"""
Test script to verify DQN setup before full training.
"""

import sys
import os

# Ensure we can import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

print("=" * 60)
print("Testing DQN Setup")
print("=" * 60)

# Test 1: Import environment with corrected imports
print("\n1. Testing environment import...")
try:
    # Fix the import path temporarily
    import src.environment.snake_game as snake_game_module
    from src.environment.snake_game import SnakeGame, Snake, Food, Vector
    print("   ✓ Snake game module imported successfully")
    
    # Now test the environment
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "snake_env", 
        "src/environment/snake_env.py"
    )
    snake_env_module = importlib.util.module_from_spec(spec)
    
    # Patch the import
    sys.modules['snake_game'] = snake_game_module
    
    spec.loader.exec_module(snake_env_module)
    SnakeEnvironment = snake_env_module.SnakeEnvironment
    
    print("   ✓ SnakeEnvironment imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Create environment
print("\n2. Testing environment creation...")
try:
    env = SnakeEnvironment(grid_size=30, render=False, seed=42)
    print("   ✓ Environment created successfully")
    print(f"   - Grid size: {env.grid_size}x{env.grid_size}")
    print(f"   - State shape: {env.get_state_shape()}")
    print(f"   - Num actions: {env.get_num_actions()}")
except Exception as e:
    print(f"   ✗ Failed to create environment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test state representation
print("\n3. Testing state representation...")
try:
    state = env.reset()
    print(f"   ✓ Environment reset successful")
    print(f"   - State shape: {state.shape}")
    print(f"   - State dtype: {state.dtype}")
    print(f"   - State features:")
    feature_names = [
        "food_dx", "food_dy",
        "danger_left", "danger_front", "danger_right",
        "dir_left", "dir_right", "dir_up", "dir_down",
        "length_norm"
    ]
    for i, (name, value) in enumerate(zip(feature_names, state)):
        print(f"     [{i}] {name:12s} = {value:6.3f}")
    
    if state.shape[0] != 10:
        print(f"   ✗ Expected 10 features, got {state.shape[0]}")
        sys.exit(1)
    print(f"   ✓ State has correct shape (10 features)")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test environment step
print("\n4. Testing environment steps...")
try:
    actions = ["LEFT", "RIGHT", "UP", "DOWN"]
    for i in range(4):
        state = env.reset()
        next_state, reward, done = env.step(i)
        print(f"   ✓ Action {i} ({actions[i]:5s}): reward={reward:+.1f}, done={done}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test random episode
print("\n5. Testing random episode...")
try:
    env.reset()
    total_reward = 0
    steps = 0
    max_steps = 200
    
    while steps < max_steps:
        action = np.random.randint(0, 4)
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    print(f"   ✓ Random episode completed")
    print(f"   - Steps: {steps}")
    print(f"   - Total reward: {total_reward}")
    print(f"   - Final score: {env.get_score()}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: PyTorch availability
print("\n6. Checking PyTorch...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   ✓ PyTorch available")
    print(f"   - Device: {device}")
    print(f"   - Version: {torch.__version__}")
except Exception as e:
    print(f"   ✗ PyTorch issue: {e}")
    sys.exit(1)

# Test 7: Config
print("\n7. Testing configuration...")
try:
    import src.config as config
    print("   ✓ Config loaded")
    print(f"   - Episodes: {config.TOTAL_EPISODES}")
    print(f"   - Batch size: {config.BATCH_SIZE}")
    print(f"   - Learning rate: {config.LEARNING_RATE}")
    print(f"   - Gamma: {config.GAMMA}")
    print(f"   - Epsilon: {config.EPSILON_START} → {config.EPSILON_END}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Replay buffer
print("\n8. Testing replay buffer...")
try:
    from src.replay_buffer import ReplayBuffer
    buffer = ReplayBuffer(capacity=100)
    
    for i in range(50):
        buffer.add(
            state=np.random.randn(11).astype(np.float32),
            action=np.random.randint(0, 4),
            reward=float(np.random.randn()),
            next_state=np.random.randn(11).astype(np.float32),
            done=False
        )
    
    print(f"   ✓ Buffer created and populated ({len(buffer)} experiences)")
    
    states, actions, rewards, next_states, dones = buffer.sample(10)
    print(f"   ✓ Sampling successful: batch shapes {states.shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: DQN Agent
print("\n9. Testing DQN agent...")
try:
    from src.agents.dqn_feature import DQNFeatureAgent
    
    agent = DQNFeatureAgent(
        state_size=10,
        action_size=4,
        device=device,
        seed=42
    )
    print("   ✓ Agent created")
    print(f"   - Epsilon: {agent.epsilon}")
    print(f"   - Memory capacity: {agent.memory.buffer.maxlen}")
    
    # Test action selection
    state = env.reset()
    action = agent.select_action(state, eval_mode=False)
    print(f"   ✓ Action selected (training): {action}")
    
    action = agent.select_action(state, eval_mode=True)
    print(f"   ✓ Action selected (eval): {action}")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 10: Training loop
print("\n10. Running mini training loop (20 episodes)...")
try:
    env = SnakeEnvironment(grid_size=20, render=False, seed=42)  # Smaller grid for speed
    agent = DQNFeatureAgent(state_size=10, action_size=4, device=device, seed=42)
    
    scores = []
    for episode in range(20):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()
            state = next_state
            steps += 1
        
        agent.update_epsilon()
        scores.append(env.get_score())
        
        if (episode + 1) % 5 == 0:
            avg_score = np.mean(scores[-5:])
            print(f"   Episode {episode + 1:2d}: score={env.get_score()}, "
                  f"steps={steps:3d}, eps={agent.epsilon:.2f}, "
                  f"avg_score={avg_score:.1f}")
    
    print(f"   ✓ Mini training completed")
    print(f"   - Average score: {np.mean(scores):.2f}")
    print(f"   - Max score: {max(scores)}")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nSetup verified. Ready for full training!")
print("\nNext steps:")
print("  1. Run: python src/train_dqn_feature.py")
print("  2. Monitor results in: results/dqn_feature/")
print("=" * 60)