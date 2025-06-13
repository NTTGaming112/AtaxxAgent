#!/usr/bin/env python3

import asyncio
import platform

from game_ui import main

if __name__ == "__main__":
    if platform.system() == "Emscripten":
        asyncio.ensure_future(main())
    else:
        asyncio.run(main())
