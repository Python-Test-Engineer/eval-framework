"""A simple example of tests using pytest"""

# By Nick from CoffeeBeforeArch

import pytest


# Simple function that squares a number
def square(num: int) -> int:
    """Doc"""
    return num * num


# Simple function that cubes another
def cube(num: int) -> int:
    """Doc"""
    return square(num) * num


# Using pytest.config for global values
def test_0211_GEN_square():

    num = 5
    result = square(num)
    assert result == num**2


# Define another test in the same file
def test_0212_GEN_1_cube():
    num = 5
    result = cube(num)
    assert result == num**3
