def example_function(column_name: str) -> str:
    """Lower case your input string.

    Args:
        column_name: Column name to transform to lower case.

    Returns:
        The lower-cased column name.
    """
    return column_name.lower()


class ExampleClass:
    """Awesome class."""

    def __init__(self, value: int):
        print("initializing")

    def bar(self) -> str:
        """Bar function."""
        return "hello"

    @property
    def foo(self) -> str:
        """Foo property."""
        return "hello"
