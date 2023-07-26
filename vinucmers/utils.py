import logging
import random


def create_logger(logger_name: str, log_file: str = ''):
    """ Creates logger.

    Args:
        logger_name: name of logger
        log_file: path to log file

    Returns:
        logger
    """
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
    """ Sets formatter.

    Args:
        handler: handler

    Returns:
        handler
    """
    formatter = logging.Formatter('[%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] [%(asctime)s] - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    return handler


def sample_substring(data, min_length, max_length, seed=42):
    """
    Sample a random substring of specified length from the given string.

    This function selects a substring of random length within the specified range
    from the provided string data.

    Parameters:
        data (str): The original string data from which the substring will be sampled.
        min_length (int): The minimum length of the substring to be sampled.
        max_length (int): The maximum length of the substring to be sampled.
        seed (int): The seed to be used for the random number generator.

    Returns:
        str: A randomly sampled substring of the given string data.
        
    Raises:
        ValueError: If the max_length is greater than the length of the original string data.
        ValueError: If the min_length is greater than the max_length.

    Example:
        >>> long_string = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        >>> sampled_substring = sample_substring(long_string, 5, 15)
        >>> print("Sampled substring:", sampled_substring)
    """
    if min_length > max_length:
        raise ValueError("The min_length should be less than or equal to the max_length.")
    
    max_length = min(max_length, len(data))
    random.seed(seed)
    length = random.randint(min_length, max_length)
    start = random.randint(0, len(data) - length)
    end = start + length

    return data[start:end]