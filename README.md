# 🐍 Mamba Sentiment Analysis Pipeline (MLOps)

This project implements a complete **MLOps architecture** for sentiment analysis on movie reviews using the innovative **Mamba** model. The goal is to demonstrate the automation of a model's lifecycle: from training and tracking to production deployment.

## 🏗️ System Architecture

The entire infrastructure is containerized with **Docker**, ensuring total reproducibility across environments:

*   **Trainer (Backend)**: Fetches the IMDB dataset (Hugging Face), trains the Mamba model, and versions the weights using MLflow.
*   **API (FastAPI)**: Serves real-time predictions through a RESTful interface.
*   **UI (Streamlit)**: A sleek, intuitive frontend for users to interact with the model instantly.

## 🚀 Key Features

- **Mamba Architecture**: Utilizes State Space Models (SSM) for faster inference compared to traditional Transformers.
- **MLflow Lifecycle**: Real-time tracking of training metrics (Loss, Accuracy) and model archiving.
- **Continuous Deployment (CD)**: The API automatically reloads the latest model weights upon successful training completion via shared Docker volumes.
- **Zero-Local Dependency**: Everything runs inside Docker—no need to install Python or PyTorch on the host machine.

## 🛠️ How to Run

### 1. Clone the repository
```bash
git clone <your-github-link>
cd mlops-mamba
