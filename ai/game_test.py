#!/usr/bin/env python3
"""
Legacy entry point for Ataxx Tournament
Redirects to modular components: tournament_runner.py and game_ui.py

This file maintains backward compatibility with the original game_test.py interface
while directing execution to the new modular architecture.
"""

import asyncio
import platform

# Import from the new modular system
from game_ui import main

if __name__ == "__main__":
    # Simply redirect to the new main function
    if platform.system() == "Emscripten":
        asyncio.ensure_future(main())
    else:
        asyncio.run(main())
