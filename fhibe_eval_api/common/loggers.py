# Copyright (c) Sony AI Inc.
# All rights reserved.
"""Module containing logger utils.

This module contains logging utilities.
"""

import logging
from typing import Optional


########################################
## Colored Console output for Logging ##
########################################
class Colors:  # noqa: D101
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[96m"
    PURPLE = "\033[95m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    RESET = "\033[0m"


class ColorFormatter(logging.Formatter):  # noqa: D101
    STRING_FORMAT = "[%(levelname)s]:%(filename)s:%(lineno)d: %(message)s"
    COLOR_FORMATS = {
        logging.DEBUG: Colors.GRAY + STRING_FORMAT + Colors.RESET,
        logging.INFO: Colors.BLUE + STRING_FORMAT + Colors.RESET,
        logging.WARNING: Colors.YELLOW + STRING_FORMAT + Colors.RESET,
        logging.ERROR: Colors.RED + STRING_FORMAT + Colors.RESET,
        logging.CRITICAL: Colors.RED + STRING_FORMAT + Colors.RESET,
    }

    def format(self, record):  # noqa: D102
        log_fmt = self.COLOR_FORMATS.get(record.levelno, self.STRING_FORMAT)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class ColoredStreamHander(logging.StreamHandler):  # noqa: D101
    def __init__(self, *args, **kwargs):  # noqa: D107
        super().__init__(*args, **kwargs)
        self.formatter = ColorFormatter()


###########################################
## All-in-one Setup function for Logging ##
###########################################
def setup_logging(level: Optional[str | int] = "debug"):  # noqa: D103
    logging_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critial": logging.CRITICAL,
    }
    if isinstance(level, str):
        level = logging_levels.get(level.lower(), logging.DEBUG)

    logging.basicConfig(
        # format="[%(levelname)s]:%(filename)s:%(lineno)d: %(message)s",
        level=level,
        handlers=[ColoredStreamHander()],
    )
