# Shared Feature Schema

This folder defines the contract between the DSA/code-analysis module and Hajera's ML module.

- `feature_schema.py`: input feature names and basic validation.
- `FEATURE_SCHEMA.md`: field meanings and coordination rules.
- `example_features.json`: example input and output records.
- `tests/test_feature_schema.py`: basic validation tests.

**Important:** `complexity_class` and `target` are training labels, not input features.
`risk_probability` is a model output, not an input feature. Static metrics estimate
complexity and defect risk; they cannot guarantee correct Big-O or prove a defect.
