import logging

_fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def get_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(_fmt)
        logger.addHandler(console)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(_fmt)
        logger.addHandler(fh)
    return logger
