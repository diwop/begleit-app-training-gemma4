"""
Python startup hook (sitecustomize) for evaluation environment.
"""

import os
import sys

# Ensure unbuffered standard output for real-time progress logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

