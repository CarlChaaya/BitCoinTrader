# Bitcoin RL Trader

A reinforcement-learning trading agent for Bitcoin built with Stable Baselines3. The project scrapes OHLCV data, trains a DDPG agent, and exposes scripts for paper-trading through the Alpaca brokerage API.

## Highlights
- Clean separation between data ingestion (`getdata.py`) and agent lifecycle (`main.py`)
- Supports historical backtesting and live paper trading
- Interactive Jupyter notebooks (`BitcoinLiveDeployment.ipynb`) for experimentation and visualisation
- Deployable to Heroku via the provided `Procfile`

## Setup
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Set the following environment variables before launching live trading:
- `APCA_API_KEY_ID`
- `APCA_API_SECRET_KEY`
- `APCA_API_BASE_URL`

Run backtests locally:
```bash
python main.py --mode backtest --starting-cash 10000
```

## Project Roadmap
- [x] Historical data ingestion and feature engineering
- [x] DDPG training pipeline and model persistence
- [ ] Automated hyper-parameter sweeps and experiment tracking
- [ ] Risk management overlays (stop loss / take profit policies)
- [ ] Docker image and CI/CD deployment scripts

Model artefacts are published separately; download the latest release assets before evaluation.
