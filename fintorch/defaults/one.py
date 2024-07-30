import os


# API and Exchanges
EXCHANGE_NAME = "binance"

# Application
MAX_TASKS = 4
MAX_WORKERS = os.cpu_count()
EXECUTOR_TYPE = "process"

__all__ = ["EXCHANGE_NAME", "MAX_TASKS", "MAX_WORKERS", "EXECUTOR_TYPE"]
