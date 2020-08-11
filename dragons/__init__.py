#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = "0.2.2"

from . import munge
from . import meraxes
from . import nbody
from . import plotutils

import logging
import coloredlogs

coloredlogs.install(
    logger=logging.getLogger("dragons"),
    fmt="%(asctime)s,%(msecs)03d %(hostname)s %(name)s %(levelname)s %(message)s",
)
