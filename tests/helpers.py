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
