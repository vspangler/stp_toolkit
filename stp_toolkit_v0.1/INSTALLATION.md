# Installation Guide for Stock Trend Prediction Toolkit

This guide provides step-by-step instructions for installing and setting up the Stock Trend Prediction Toolkit on your system.

## System Requirements

- **Python Version**: Python 3.8 or newer
- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 4GB RAM recommended
- **Storage**: At least 1GB free space for dependencies and data

## Prerequisites

### 1. Python Installation

Ensure you have Python 3.8+ installed on your system:

```bash
python --version
```

If Python is not installed or the version is older than 3.8:
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: Use Homebrew: `brew install python` or download from [python.org](https://www.python.org/downloads/)
- **Linux**: Use your package manager: `sudo apt install python3 python3-pip` (Ubuntu/Debian)

### 2. pip Package Manager

Verify pip is installed:

```bash
pip --version
```

If pip is not available, install it:
```bash
python -m ensurepip --upgrade
```

## Installation Steps

### Step 1: Download the Project

Clone or download the repository to your local machine:

```bash
git clone <repository-url>
cd stp_toolkit_v0.1
```

Or download and extract the ZIP file to your desired directory.

### Step 2: Create Virtual Environment (Recommended)

Create and activate a virtual environment to isolate project dependencies:

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies

Install all required Python packages:

```bash
pip install yfinance pandas numpy scikit-learn tensorflow
```

#### Individual Package Information:
- **yfinance**: For fetching stock market data from Yahoo Finance
- **pandas**: For data manipulation and analysis
- **numpy**: For numerical computations
- **scikit-learn**: For machine learning algorithms
- **tensorflow**: For deep learning (LSTM) models

### Step 4: Verify Installation

Test the installation by running a simple import check:

```python
python -c "import yfinance, pandas, numpy, sklearn, tensorflow; print('All dependencies installed successfully!')"
```

## Platform-Specific Configuration

### Windows Long Path Support (If Needed)

If you encounter path length issues on Windows:

1. Open Start menu and type "regedit" (Registry Editor)
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Set `LongPathsEnabled` to `1`
4. Restart your computer

### GPU Support for TensorFlow (Optional)

For faster deep learning computations, you can install TensorFlow with GPU support:

1. Install CUDA toolkit (version compatible with TensorFlow)
2. Install cuDNN
3. Install TensorFlow GPU version:
   ```bash
   pip install tensorflow-gpu
   ```

## Quick Start Test

Test your installation by running one of the main scripts:

```bash
python main.py
```

When prompted, enter a stock symbol (e.g., `AAPL`) to verify everything works correctly.

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
**Problem**: `ModuleNotFoundError` when running scripts
**Solution**: 
- Ensure virtual environment is activated
- Reinstall missing packages: `pip install <package-name>`

#### 2. TensorFlow Installation Issues
**Problem**: TensorFlow fails to install or import
**Solution**:
- Update pip: `pip install --upgrade pip`
- Install specific TensorFlow version: `pip install tensorflow==2.12.0`
- For older systems, try: `pip install tensorflow-cpu`

#### 3. Data Fetching Errors
**Problem**: yfinance fails to fetch data
**Solution**:
- Check internet connection
- Verify stock symbol is valid
- Try again later (Yahoo Finance may have temporary issues)

#### 4. Memory Issues
**Problem**: Out of memory errors during execution
**Solution**:
- Close other applications
- Use smaller date ranges for analysis
- Consider using a machine with more RAM

### Getting Help

If you encounter issues not covered here:

1. Check if all dependencies are properly installed
2. Ensure you're using Python 3.8+
3. Try running with a different stock symbol
4. Check the project's README.md for additional information

## Updating Dependencies

To update all packages to their latest versions:

```bash
pip install --upgrade yfinance pandas numpy scikit-learn tensorflow
```

## Uninstalling

To remove the project and its dependencies:

1. Deactivate virtual environment: `deactivate`
2. Remove project directory
3. Delete virtual environment folder

## Next Steps

After successful installation:

1. Read the [README.md](README.md) for usage instructions
2. Run `python main.py` for integrated forecasting
3. Try `python ai_estimate.py` for AI-driven trend prediction
4. Explore `python stock_estimate.py` for comprehensive technical analysis

## Project Structure

```
stp_toolkit_v0.1/
├── main.py              # Main orchestrator script
├── ai_estimate.py       # AI-driven trend prediction
├── stock_estimate.py    # Comprehensive technical analysis
├── README.md           # Project documentation
├── INSTALLATION.md     # This installation guide
├── CHANGELOG.md        # Version history
└── DISCLAIMER.md       # Legal disclaimer
```

---

**Note**: This toolkit is for educational and research purposes. Always consult with financial professionals before making investment decisions based on these predictions.
