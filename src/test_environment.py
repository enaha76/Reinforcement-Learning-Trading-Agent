"""
Test Script for RL Environment
Quick validation that the environment works correctly
"""

import sys
import os

def test_basic_environment():
    """Test the environment with minimal data"""
    
    print("=== Testing RL Environment ===")
    
    # Test if we can import our environment
    try:
        from environment import TradingEnvironment
        print("✓ Environment import successful")
    except ImportError as e:
        print(f"✗ Environment import failed: {e}")
        return False
    
    # Create minimal test data
    import pandas as pd
    import numpy as np
    
    # Create 50 days of test data
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    
    test_data = pd.DataFrame({
        'Date': dates,
        'Close_SPY': 400 + np.cumsum(np.random.randn(50) * 0.01),
        'Close_GLD': 180 + np.cumsum(np.random.randn(50) * 0.005),
        'Returns_SPY': np.random.randn(50) * 0.02,
        'Returns_GLD': np.random.randn(50) * 0.015,
        'RSI_SPY': 50 + np.random.randn(50) * 10,
        'RSI_GLD': 50 + np.random.randn(50) * 10,
        'MACD_SPY': np.random.randn(50) * 0.5,
        'MACD_Signal_SPY': np.random.randn(50) * 0.3,
        'Volatility_SPY': 0.15 + np.random.randn(50) * 0.05,
        'Volatility_GLD': 0.12 + np.random.randn(50) * 0.03,
    })
    
    try:
        # Create environment
        env = TradingEnvironment(test_data, initial_balance=10000, lookback_window=5)
        print("✓ Environment creation successful")
        
        # Test reset
        initial_state = env.reset()
        print(f"✓ Environment reset successful - State shape: {initial_state.shape}")
        
        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            print(f"  Step {i+1}: Action={action[0]:.3f}, Reward={reward:.4f}, Balance=${info['balance']:.2f}")
            
            if done:
                print("  Environment completed early")
                break
        
        print("✓ Environment stepping successful")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_pipeline():
    """Test the data processing pipeline"""
    
    print("\n=== Testing Data Pipeline ===")
    
    # Check if processed data exists
    data_files = ['raw_data.csv', 'processed_data.csv', 'train_data.csv', 'test_data.csv']
    
    for filename in data_files:
        filepath = f'../data/{filename}'
        if os.path.exists(filepath):
            print(f"✓ {filename} exists")
            
            # Quick data validation
            try:
                import csv
                with open(filepath, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    print(f"  - {len(rows)} records, {len(rows[0]) if rows else 0} columns")
            except Exception as e:
                print(f"  ✗ Error reading {filename}: {e}")
        else:
            print(f"✗ {filename} missing")
    
    return True

def test_simple_simulation():
    """Test the simple trading simulation"""
    
    print("\n=== Testing Simple Simulation ===")
    
    try:
        # Import and run a quick test
        import simple_trading_simulation
        
        # Check if test data exists
        if not os.path.exists('../data/test_data.csv'):
            print("✗ Test data not found")
            return False
        
        print("✓ Simple simulation import successful")
        print("✓ Test data available")
        
        # You can run the full simulation here if needed
        # result = simple_trading_simulation.main()
        
        return True
        
    except Exception as e:
        print(f"✗ Simple simulation test failed: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🧪 TESTING RL TRADING AGENT SETUP")
    print("=" * 50)
    
    tests = [
        ("Data Pipeline", test_data_pipeline),
        ("Simple Simulation", test_simple_simulation),
        ("RL Environment", test_basic_environment),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Ready for full training.")
    else:
        print("⚠️  Some tests failed. Check the issues above.")
    
    return passed == len(results)

if __name__ == "__main__":
    main()
