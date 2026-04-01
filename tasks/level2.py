"""Level 2: Resource Leak (Medium)"""

DESCRIPTION = """Fix an unclosed file handle with a context manager.

The read_config function opens a file using open() but never closes
the file handle. This can lead to resource exhaustion and file locking
issues in long-running applications.

Your task is to fix the function using Python's context manager (with statement)
to ensure the file is properly closed after reading.
"""

BUGGY_CODE = '''def read_config(filepath):
    """Read configuration from a file."""
    f = open(filepath, "r")  # Bug: file handle is never closed
    content = f.read()
    return content
'''

CORRECT_CODE = '''def read_config(filepath):
    """Read configuration from a file."""
    with open(filepath, "r") as f:
        return f.read()
'''
