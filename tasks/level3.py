"""Level 3: Race Condition (Hard)

A multi-threaded counter that is not thread-safe.
The agent must add proper locking to prevent race conditions.
"""

DESCRIPTION = """Fix thread-unsafe counter with a Lock.

This code creates multiple threads that all increment a shared counter.
However, the counter increment operation is not atomic, leading to a race
condition where the final count is less than expected.

Your task is to add proper thread synchronization using threading.Lock()
to ensure the counter reaches the correct value (50000 with 5 threads
each incrementing 10000 times).
"""

BUGGY_CODE = '''import threading

counter = 0

def increment():
    """Increment the global counter 10000 times."""
    global counter
    for _ in range(10000):
        counter += 1  # Bug: not thread-safe


# Create 5 threads that all increment the counter
threads = [threading.Thread(target=increment) for _ in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(counter)  # Should be 50000, but is often less due to race condition
'''

CORRECT_CODE = '''import threading

counter = 0
lock = threading.Lock()

def increment():
    """Increment the global counter 10000 times, thread-safely."""
    global counter
    for _ in range(10000):
        with lock:
            counter += 1


# Create 5 threads that all increment the counter
threads = [threading.Thread(target=increment) for _ in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(counter)  # Now correctly prints 50000
'''
