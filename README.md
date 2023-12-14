# cryptoforecaster
## Recurrent Neural Network for Cryptocurrency Return Forecasting
This is a minimal version of [trading-ml](https://github.com/domreichl/trading-ml), to be used for tuning and testing binary sign predictor RNNs that forecast cryptocurrency returns.

### Installation
1. Install virtual environment: `python -m venv .venv`
2. Activate virtual evironment:
    - Linux: `source .venv/bin/activate`
    - Windows: `.venv\Scripts\activate.bat`
3. Install piptools: `python -m pip install pip-tools`
4. Compile project: `python -m piptools compile pyproject.toml`
5. Install project: `python -m pip install -e .`

### Commands
1. Download new data: `cf download`
2. Update current data: `cf update`
3. Tune the model: `cf tune`
4. Test the model: `cf test`
5. Plot test results: `cf plot test`
