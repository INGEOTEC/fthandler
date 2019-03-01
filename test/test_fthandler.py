import os

try:
    os.mkdir("tmp")
except OSError:
    pass
