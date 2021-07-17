# Getting full or absolute path of the current directory

# Importing needed library
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

print(current_dir)
