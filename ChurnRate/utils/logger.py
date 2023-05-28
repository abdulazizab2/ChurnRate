import logging

logging.basicConfig(
    filename="logs/logs.log",
    filemode='w',
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s"
)
