#!/usr/bin/env python3
"""
Launcher script for Mokuro GUI
"""

import sys
import os
from pathlib import Path

# Add mokuro to path
mokuro_path = Path(__file__).parent / 'mokuro'
sys.path.insert(0, str(mokuro_path))

# Import and run the GUI
from mokuro.mokuro_gui import main

if __name__ == '__main__':
    main()
