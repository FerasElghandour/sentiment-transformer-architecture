# Validation

Run `python smoke_test.py` for the focused offline behavior checks. Fixtures are generated locally. These checks do not train a model, access private data, call a model provider, or recertify historical metrics.

## Executed checks — October 17, 2026

Causal classifier shapes, right-padding invariance, finite gradients, and padding guards passed.

Passed on Python 3.14.3; the workflow repeats these checks on Python 3.12.
