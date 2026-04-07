"""
Test suite for Email Triage Environment
Validates OpenEnv spec compliance and environment functionality
"""

import sys
import json
from env import EmailTriageEnv, Action, Category, Priority
from pydantic import BaseModel, ValidationError


def test_models():
    """Test Pydantic models"""
    print("[TEST] Validating Pydantic models...")
    
    try:
        # Test Action model
        action = Action(
            category=Category.SUPPORT,
            priority=Priority.HIGH,
            response_required=True
        )
        assert action.category == Category.SUPPORT
        assert action.priority == Priority.HIGH
        assert action.response_required is True
        print("  ✓ Action model valid")
        
        # Test invalid category
        try:
            bad_action = Action(
                category="invalid",
                priority=Priority.HIGH,
                response_required=True
            )
            print("  ✗ Model validation failed - should reject invalid category")
            return False
        except (ValueError, ValidationError):
            print("  ✓ Model validation works (rejects invalid)")
            
        return True
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False


def test_environment_init():
    """Test environment initialization"""
    print("[TEST] Testing environment initialization...")
    
    try:
        env_easy = EmailTriageEnv(task="easy")
        env_medium = EmailTriageEnv(task="medium")
        env_hard = EmailTriageEnv(task="hard")
        
        assert env_easy.task == "easy"
        assert env_medium.task == "medium"
        assert env_hard.task == "hard"
        assert env_easy.max_steps == 10
        
        print("  ✓ Environment initialization works")
        return True
    except Exception as e:
        print(f"  ✗ Environment init failed: {e}")
        return False


def test_reset():
    """Test reset functionality"""
    print("[TEST] Testing reset()...")
    
    try:
        env = EmailTriageEnv(task="easy")
        result = env.reset()
        
        # Validate result structure
        assert hasattr(result, 'observation')
        assert hasattr(result, 'info')
        assert result.observation is not None
        
        # Validate observation fields
        obs = result.observation
        assert obs.email_id is not None
        assert obs.sender is not None
        assert obs.subject is not None
        assert obs.body_preview is not None
        assert obs.has_attachment is not None
        assert obs.current_step == 0
        assert obs.total_emails_in_batch > 0
        
        print(f"  ✓ Reset works (batch size: {obs.total_emails_in_batch})")
        return True
    except Exception as e:
        print(f"  ✗ Reset test failed: {e}")
        return False


def test_step():
    """Test step functionality"""
    print("[TEST] Testing step()...")
    
    try:
        env = EmailTriageEnv(task="easy")
        env.reset()
        
        action = Action(
            category=Category.PROMOTIONAL,
            priority=Priority.LOW,
            response_required=False
        )
        
        result = env.step(action)
        
        # Validate result structure
        assert hasattr(result, 'observation')
        assert hasattr(result, 'reward')
        assert hasattr(result, 'done')
        assert hasattr(result, 'info')
        
        # Validate reward range
        assert 0.0 <= result.reward <= 1.0, f"Reward out of range: {result.reward}"
        
        # Validate observation
        obs = result.observation
        assert obs.current_step >= 0
        
        # Validate info dict
        assert 'email_id' in result.info
        assert 'category_correct' in result.info
        assert 'priority_correct' in result.info
        assert 'response_correct' in result.info
        
        print(f"  ✓ Step works (reward: {result.reward:.2f}, done: {result.done})")
        return True
    except Exception as e:
        print(f"  ✗ Step test failed: {e}")
        return False


def test_episode():
    """Test complete episode"""
    print("[TEST] Testing complete episode...")
    
    try:
        env = EmailTriageEnv(task="medium")
        result = env.reset()
        
        total_reward = 0.0
        steps = 0
        
        while not result.done and steps < env.max_steps:
            # Agent takes a random action
            action = Action(
                category=Category.INTERNAL,
                priority=Priority.MEDIUM,
                response_required=True
            )
            
            result = env.step(action)
            total_reward += result.reward
            steps += 1
        
        state = env.state()
        
        # Validate state
        assert state['current_step'] == steps
        assert state['batch_size'] > 0
        assert abs(state['total_reward'] - total_reward) < 0.01
        
        print(f"  ✓ Episode completed ({steps} steps, reward: {total_reward:.2f})")
        return True
    except Exception as e:
        print(f"  ✗ Episode test failed: {e}")
        return False


def test_task_difficulties():
    """Test all three task difficulties"""
    print("[TEST] Testing task difficulties...")
    
    try:
        tasks = ["easy", "medium", "hard"]
        for task in tasks:
            env = EmailTriageEnv(task=task)
            result = env.reset()
            
            batch_size = result.observation.total_emails_in_batch
            assert batch_size > 0
            print(f"  ✓ {task.capitalize()} task: {batch_size} emails")
        
        return True
    except Exception as e:
        print(f"  ✗ Difficulty test failed: {e}")
        return False


def test_reward_range():
    """Test reward values stay in [0, 1] range"""
    print("[TEST] Testing reward range...")
    
    try:
        env = EmailTriageEnv(task="hard")
        env.reset()
        
        rewards = []
        for _ in range(env.max_steps):
            action = Action(
                category=Category.SPAM,
                priority=Priority.LOW,
                response_required=False
            )
            result = env.step(action)
            rewards.append(result.reward)
            
            if result.done:
                break
        
        # Verify all rewards in range
        for r in rewards:
            assert 0.0 <= r <= 1.0, f"Invalid reward: {r}"
        
        print(f"  ✓ All rewards in [0, 1]: min={min(rewards):.2f}, max={max(rewards):.2f}")
        return True
    except Exception as e:
        print(f"  ✗ Reward range test failed: {e}")
        return False


def test_yaml_format():
    """Test openenv.yaml is well-formed"""
    print("[TEST] Validating openenv.yaml...")
    
    try:
        import yaml
        with open('openenv.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required = ['name', 'version', 'tasks', 'action_space', 'observation_space']
        for field in required:
            assert field in config, f"Missing required field: {field}"
        
        # Check tasks
        assert len(config['tasks']) >= 3, "Must have at least 3 tasks"
        
        print("  ✓ openenv.yaml is valid")
        return True
    except ImportError:
        print("  ⚠ yaml not installed, skipping YAML validation")
        return True
    except Exception as e:
        print(f"  ✗ YAML validation failed: {e}")
        return False


def test_deterministic():
    """Test that environment is deterministic"""
    print("[TEST] Testing deterministic behavior...")
    
    try:
        # Run same actions twice
        rewards_1 = []
        env1 = EmailTriageEnv(task="easy")
        env1.reset()
        
        actions = [
            Action(category=Category.PROMOTIONAL, priority=Priority.LOW, response_required=False),
            Action(category=Category.SUPPORT, priority=Priority.HIGH, response_required=True),
            Action(category=Category.SPAM, priority=Priority.LOW, response_required=False),
        ]
        
        for action in actions:
            result = env1.step(action)
            rewards_1.append(result.reward)
        
        # Second run
        rewards_2 = []
        env2 = EmailTriageEnv(task="easy")
        env2.reset()
        
        for action in actions:
            result = env2.step(action)
            rewards_2.append(result.reward)
        
        # Compare
        assert len(rewards_1) == len(rewards_2)
        for r1, r2 in zip(rewards_1, rewards_2):
            assert abs(r1 - r2) < 1e-6, f"Rewards differ: {r1} vs {r2}"
        
        print("  ✓ Environment is deterministic")
        return True
    except Exception as e:
        print(f"  ✗ Determinism test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 70)
    print("EMAIL TRIAGE ENVIRONMENT - COMPLIANCE TESTS")
    print("=" * 70)
    print()
    
    tests = [
        test_models,
        test_environment_init,
        test_reset,
        test_step,
        test_episode,
        test_task_difficulties,
        test_reward_range,
        test_yaml_format,
        test_deterministic,
    ]
    
    results = []
    for test_func in tests:
        try:
            passed = test_func()
            results.append((test_func.__name__, passed))
        except Exception as e:
            print(f"[ERROR] Uncaught exception in {test_func.__name__}: {e}")
            results.append((test_func.__name__, False))
        print()
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, passed_flag in results:
        status = "✓ PASS" if passed_flag else "✗ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Result: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
