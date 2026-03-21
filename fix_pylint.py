import os
import re

def fix_file(filepath, fixes):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for ln, pattern, replacement in fixes:
        if ln < len(lines):
            lines[ln] = re.sub(pattern, replacement, lines[ln])
    with open(filepath, 'w') as f:
        f.writelines(lines)

# Example fixes tuple format: (line_index_0_based, regex_pattern, replacement)
