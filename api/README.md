# Trading Bot API

This directory contains the FastAPI backend that serves data to the Quant-Dash dashboard. It provides endpoints to access trading performance, history, and real-time signals.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- An active virtual environment is highly recommended.

### Installation

1.  **Navigate to the project root directory:**
    ```bash
    cd /path/to/your/Trading_Bot
    ```

2.  **Install the required dependencies:**
    All dependencies for this API are included in the main `requirements.txt` file in the project root.
    ```bash
    pip install -r requirements.txt
    ```

### Running the API

1.  From the project root directory, you can run the API using the provided batch script:
    ```bash
    ./run_api.bat
    ```
    Or, you can run it directly with uvicorn:
    ```bash
    uvicorn api.main:app --reload
    ```

2.  The API will be available at `http://localhost:8000`.

## Endpoints

The API exposes several routers to organize endpoints:

-   `/dashboard`: General dashboard data.
-   `/performance`: Detailed performance metrics.
-   `/win-loss`: Win/loss analysis data.
-   `/history`: Historical trade data.
-   `/active-trades`: Information on currently active trades.
-   `/signals`: Real-time trading signals.

You can explore all available endpoints and interact with them via the auto-generated Swagger documentation at `http://localhost:8000/docs`. 