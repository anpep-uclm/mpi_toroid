#!/usr/bin/env python3
import random
import sys

print(','.join([str(random.uniform(-1e6, 1e6)) for _ in range(0, 1 if len(sys.argv) < 2 else int(sys.argv[1]))]))
