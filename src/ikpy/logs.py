# coding= utf8
import logging

logger = logging.getLogger("ikpy")
stream_handler = logging.StreamHandler()
logger.setLevel(logging.WARNING)
logger.addHandler(stream_handler)


def set_log_level(level):
    logger.setLevel(level)


