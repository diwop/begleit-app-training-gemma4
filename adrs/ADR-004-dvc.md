# ADR-004 DVC to store training data

Training data is stored using Data Version Control.
The actual data is not stored in the repository but in a separate store.

## Rationale

The training data is quite large and not free to be published on Github.
The repository therfore just contains DVC pointers to the training data which resides in S3.