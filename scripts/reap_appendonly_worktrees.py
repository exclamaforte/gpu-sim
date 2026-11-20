#!/usr/bin/env python3
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
raise SystemExit(subprocess.call([sys.executable, os.path.join(here, "appendonly_worktrees.py"), "reap"] + sys.argv[1:]))
