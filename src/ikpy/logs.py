# coding= utf8
import logging

manager = logging.getLogger()
manager.setLevel(logging.ERROR)
stream_handler = logging.StreamHandler()
manager.addHandler(stream_handler)
