"""Level 1: KeyError Bug (Easy)"""

DESCRIPTION = """Fix a function that crashes on missing dict keys.

The get_user_score function directly accesses dictionary keys without
checking if they exist. This causes a KeyError when the user_id is not
in the data dictionary.

Your task is to fix the function so it handles missing keys gracefully
and returns a default value (0) instead of crashing.
"""

BUGGY_CODE = '''def get_user_score(data, user_id):
    """Get a user's score from the data dictionary."""
    return data[user_id]["score"]  # Bug: no guard for missing key


# Test the function
test_data = {"alice": {"score": 95}, "bob": {"score": 87}}
print(get_user_score(test_data, "charlie"))  # This crashes with KeyError
'''

CORRECT_CODE = '''def get_user_score(data, user_id):
    """Get a user's score from the data dictionary."""
    return data.get(user_id, {}).get("score", 0)


# Test the function
test_data = {"alice": {"score": 95}, "bob": {"score": 87}}
print(get_user_score(test_data, "charlie"))  # Returns 0 instead of crashing
'''
