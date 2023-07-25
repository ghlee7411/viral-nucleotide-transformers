import logging


def create_logger(logger_name: str, log_file: str = ''):
    parent_logger_name = logger_name.rsplit('.', 1)[0]
    if logger_name != parent_logger_name:
        parent_logger = logging.getLogger(parent_logger_name)

        if len(parent_logger.handlers) > 0:
            return logging.getLogger(logger_name)

    logger = logging.getLogger(logger_name)

    if len(logger.handlers) > 0:
        return logger

    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler = set_formatter(stream_handler)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler = set_formatter(file_handler)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def set_formatter(handler):
    formatter = logging.Formatter('📃 [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] [%(asctime)s] - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    return handler
