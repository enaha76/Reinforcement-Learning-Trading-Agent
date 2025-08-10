"""
Simple Validation Script
Tests the setup without external dependencies
"""

import os
import csv

def check_data_files():
    """Check if all data files exist and are valid"""
    
    print("=== Checking Data Files ===")
    
    data_files = {
        'raw_data.csv': 'Raw market data',
        'processed_data.csv': 'Processed data with indicators', 
        'train_data.csv': 'Training dataset',
        'test_data.csv': 'Testing dataset'
    }
    
    all_good = True
    
    for filename, description in data_files.items():
        filepath = f'../data/{filename}'
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    
                print(f"✓ {filename}: {len(rows)} records, {len(rows[0]) if rows else 0} columns")
                
                # Check for required columns
                if filename == 'processed_data.csv' and rows:
                    required_cols = ['Date', 'Close_SPY', 'Close_GLD', 'Returns_SPY', 'Returns_GLD']
                    missing_cols = [col for col in required_cols if col not in rows[0]]
                    
                    if missing_cols:
                        print(f"  ⚠️  Missing columns: {missing_cols}")
                        all_good = False
                    else:
                        print(f"  ✓ All required columns present")
                        
            except Exception as e:
                print(f"✗ {filename}: Error reading file - {e}")
                all_good = False
        else:
            print(f"✗ {filename}: File not found")
            all_good = False
    
    return all_good

def check_scripts():
    """Check if all scripts exist"""
    
    print("\n=== Checking Scripts ===")
    
    scripts = {
        'simple_data_collection.py': 'Data collection',
        'simple_preprocessing.py': 'Data preprocessing',
        'simple_trading_simulation.py': 'Trading simulation',
        'environment.py': 'RL environment',
        'train.py': 'Training script',
        'backtest.py': 'Backtesting script'
    }
    
    all_good = True
    
    for filename, description in scripts.items():
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"✓ {filename}: {file_size} bytes - {description}")
        else:
            print(f"✗ {filename}: Missing - {description}")
            all_good = False
    
    return all_good

def test_simple_import():
    """Test importing our simple modules"""
    
    print("\n=== Testing Simple Imports ===")
    
    try:
        import simple_trading_simulation
        print("✓ simple_trading_simulation import successful")
        return True
    except Exception as e:
        print(f"✗ simple_trading_simulation import failed: {e}")
        return False

def run_quick_simulation_test():
    """Run a very quick test of the simulation"""
    
    print("\n=== Quick Simulation Test ===")
    
    try:
        # Check if we have test data
        if not os.path.exists('../data/test_data.csv'):
            print("✗ No test data available")
            return False
        
        # Import and test basic functionality
        import simple_trading_simulation
        
        # Load a small sample of data
        test_data = simple_trading_simulation.load_data('test_data.csv')
        
        if not test_data:
            print("✗ Could not load test data")
            return False
        
        print(f"✓ Loaded {len(test_data)} test records")
        
        # Test strategy creation
        strategy = simple_trading_simulation.SimpleTradingStrategy(initial_balance=10000)
        benchmark = simple_trading_simulation.BuyAndHoldStrategy(initial_balance=10000)
        
        print("✓ Strategy objects created successfully")
        
        # Test a few steps
        for i, data_point in enumerate(test_data[:5]):
            strategy.step(data_point)
            benchmark.step(data_point)
            
            if i == 4:  # After 5 steps
                print(f"✓ Simulation test completed - Strategy balance: ${strategy.balance:.2f}")
                print(f"✓ Benchmark balance: ${benchmark.balance:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Quick simulation test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    
    print("🔍 VALIDATING RL TRADING AGENT SETUP")
    print("=" * 50)
    
    tests = [
        ("Data Files", check_data_files),
        ("Scripts", check_scripts), 
        ("Simple Import", test_simple_import),
        ("Quick Simulation", run_quick_simulation_test)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED!")
        print("Your setup is working correctly and ready for:")
        print("• Full RL training (once dependencies are installed)")
        print("• Real market data collection") 
        print("• Complete backtesting and analysis")
        
        print("\nNext steps:")
        print("1. Install full dependencies: pip3 install -r ../requirements.txt")
        print("2. Run: python3 data_collection.py (for real data)")
        print("3. Run: python3 train.py (to train RL agent)")
        print("4. Run: python3 backtest.py (for full analysis)")
        
    elif passed >= len(results) - 1:
        print("\n✅ MOSTLY WORKING!")
        print("Your core setup is functional. Minor issues can be resolved.")
        
    else:
        print("\n⚠️  ISSUES DETECTED")
        print("Please address the failed tests above before proceeding.")
    
    return passed >= len(results) - 1

if __name__ == "__main__":
    main()
