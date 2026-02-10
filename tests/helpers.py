import unittest


class TestHelpers:
    """
    A utility class with common methods for tests.

    This class should be inherited from the unittest.TestCase class when creating a test.
    Example:
        import unittest
        from helpers import TestHelpers

        class TestExample(unittest.TestCase, TestHelpers):
    """

    def AssertBlank(self, function):
        """Assert that the function can be called without raising exceptions."""
        try:
            function()
        except Exception as e:
            self.fail(f"Function {function.__name__} raised an unexpected exception: {e}")

    def assetBetween(self, value, low, high, msg=""):
        """
        Assert that the provided `value` is between `low` and `high`, inclusive.

        Args:
            value: The value to test.
            low: The lower bound of the acceptable range.
            high: The upper bound of the acceptable range.

        Raises:
            AssertionError: If the `value` is not between `low` and `high`.
        """
        if not (low <= value <= high):
            base_message = f"Value {value} is not between {low} and {high}."
            full_message = f"{base_message} {msg}" if msg else base_message
            self.fail(full_message)
