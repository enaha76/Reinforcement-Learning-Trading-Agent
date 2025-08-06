# Reinforcement Learning Trading Agent

A sophisticated reinforcement learning system that dynamically allocates assets between SPY (S&P 500 ETF) and GLD (Gold ETF) to maximize risk-adjusted returns.

## 🎯 Project Objectives

- **Primary Goal**: Achieve steady 20% better performance than Buy-and-Hold SPY strategy
- **Stretch Goal**: Generate 50% more final portfolio value over 5+ years
- **Approach**: Use Sharpe ratio-based rewards to encourage risk-adjusted returns

## 🏗️ Architecture

### Core Components

- **State Space**: Current portfolio allocation + 30-day window of technical indicators
- **Action Space**: Continuous [0,1] representing SPY allocation percentage
- **Reward Function**: Sharpe ratio-based for risk-adjusted performance
- **Algorithm**: Proximal Policy Optimization (PPO) from stable-baselines3
- **Environment**: Custom gym environment built on FinRL framework

### Technical Indicators

- **MACD**: Momentum indicator
- **RSI**: Overbought/oversold conditions  
- **ATR**: Market volatility measure
- **Bollinger Bands**: Price volatility bands
- **Moving Averages**: Trend indicators

## 📁 Project Structure

```
RL-project/
├── data/
│   ├── raw_data.csv          # Historical SPY/GLD data
│   ├── processed_data.csv    # Data with technical indicators
│   ├── train_data.csv        # Training dataset (75%)
│   └── test_data.csv         # Testing dataset (25%)
├── models/
│   └── ppo_spy_gld.zip       # Trained RL model
├── notebooks/
│   └── analysis.ipynb        # Data exploration (optional)
├── results/
│   ├── performance_comparison.png
│   ├── training_progress.png
│   └── training_log.csv
├── src/
│   ├── data_collection.py    # Download market data
│   ├── data_preprocessing.py # Calculate indicators & split data
│   ├── environment.py        # Custom RL environment
│   ├── train.py             # Train the RL agent
│   └── backtest.py          # Evaluate performance
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install pip if not available
sudo apt install python3-pip -y

# Install required packages
pip3 install -r requirements.txt
```

### 2. Collect Data

```bash
cd src
python3 data_collection.py
```

This downloads SPY and GLD historical data from 2005 to present.

### 3. Preprocess Data

```bash
python3 data_preprocessing.py
```

Calculates technical indicators and splits data into train/test sets.

### 4. Train the Agent

```bash
python3 train.py
```

Trains the PPO agent using the training dataset. This may take 30-60 minutes.

### 5. Evaluate Performance

```bash
python3 backtest.py
```

Runs backtest comparison against Buy-and-Hold benchmark and generates performance plots.

## 📊 Key Features

### Environment Design
- **Realistic Trading**: Includes transaction costs and portfolio rebalancing
- **Risk Management**: Sharpe ratio rewards encourage balanced risk-return
- **Market Context**: 30-day lookback window provides trend awareness

### Model Architecture
- **PPO Algorithm**: Proven stable for continuous control problems
- **Neural Network**: 256x256 hidden layers for both policy and value functions
- **Regularization**: Entropy bonus and gradient clipping for stable training

### Performance Analysis
- **Comprehensive Metrics**: Total return, Sharpe ratio, max drawdown, win rate
- **Visual Comparisons**: Portfolio value, cumulative returns, asset allocation
- **Rolling Analysis**: Time-varying Sharpe ratio comparison

## 🎯 Expected Results

Based on the sophisticated design:

- **Sharpe Ratio**: Should exceed Buy-and-Hold due to dynamic allocation
- **Volatility**: Lower than 100% equity exposure through gold diversification  
- **Drawdowns**: Reduced through tactical allocation adjustments
- **Returns**: Target 20%+ improvement over benchmark

## 🔧 Configuration

### Environment Parameters
- `initial_balance`: Starting portfolio value (default: $100,000)
- `lookback_window`: Days of history in state (default: 30)
- `transaction_cost`: Trading cost percentage (default: 0.1%)
- `risk_free_rate`: For Sharpe ratio calculation (default: 2%)

### Training Parameters
- `total_timesteps`: Training duration (default: 50,000)
- `learning_rate`: PPO learning rate (default: 3e-4)
- `n_steps`: Steps per update (default: 2048)
- `batch_size`: Minibatch size (default: 64)

## 📈 Monitoring Training

Training progress is logged and can be monitored via:

1. **Console Output**: Real-time training statistics
2. **CSV Logs**: Detailed episode data in `results/training_log.csv`
3. **TensorBoard**: Launch with `tensorboard --logdir results/tensorboard/`
4. **Progress Plots**: Generated automatically after training

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Data Issues**: Check internet connection for yfinance downloads
3. **Memory Issues**: Reduce `n_steps` or `batch_size` if needed
4. **Training Slow**: Consider reducing `total_timesteps` for testing

### Performance Tips

- **GPU Acceleration**: Install `torch` with CUDA support if available
- **Parallel Training**: Use `n_envs > 1` for faster data collection
- **Hyperparameter Tuning**: Adjust learning rate and network architecture

## 📚 Technical Details

### State Vector Composition
```
[current_spy_allocation, current_gld_allocation, 
 feature_day_1, feature_day_2, ..., feature_day_30]
```

### Reward Calculation
```python
sharpe_ratio = (mean_return - risk_free_rate) / std_return * sqrt(252)
reward = sharpe_ratio / 10.0 + allocation_penalty
```

### Action Interpretation
```python
spy_allocation = action[0]  # [0, 1]
gld_allocation = 1.0 - spy_allocation
```

## 🎓 Learning Outcomes

This project demonstrates:

- **RL Environment Design**: Custom gym environments for financial applications
- **Feature Engineering**: Technical indicators and market data processing
- **Risk Management**: Sharpe ratio optimization vs pure return maximization
- **Backtesting**: Rigorous out-of-sample performance evaluation
- **Visualization**: Comprehensive performance analysis and plotting

## 📄 License

This project is for educational and research purposes. Not intended for actual trading without proper risk management and compliance review.

## 🤝 Contributing

Feel free to experiment with:
- Different technical indicators
- Alternative reward functions
- Multi-asset portfolios
- Different RL algorithms (SAC, TD3, etc.)

---

**Disclaimer**: This is a research project. Past performance does not guarantee future results. Always consult financial professionals before making investment decisions.

# Reinforcement-Learning-Trading-Agent