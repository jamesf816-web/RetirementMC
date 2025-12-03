import numpy as np

def inflate_amount(amount, rate, years):
    return amount * (1 + rate)**years
