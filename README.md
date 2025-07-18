# INSIGHT SUGAR

A Python implementation of a Multilayer Perceptron (MLP) from scratch for diabetes prediction using the Pima Indians Diabetes dataset.

## Prerequisites

- Python3
- pip
- virtualenv (recommended)

## Setup Instructions

### 1. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate
```

### 2. Install Requirements
Option A: Basic installation (using pip)
```bash
pip install -r requirements.txt
```

Option B: Advanced installation (using pip-tools)
1. First install pip-tools:
   
```bash
pip install pip-tools
```
2. Compile requirements from `requirements.in`:
```bash
pip-compile requirements.in
```
3. Sync your virtual environment:
```bash
pip-sync requirements.txt
```

## Running the Project
### 1. Activate your virtual environment (if not already active):
```bash
source venv/bin/activate # On Windows use `venv\Scripts\activate`
```
### 2. Run the main script
```bash
python main.py
```
