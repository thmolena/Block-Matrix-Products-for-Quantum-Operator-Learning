# External matrix registry

The study uses the PARSEC group of the SuiteSparse Matrix Collection. Matrix archives are not redistributed. `checksums.sha256` pins the four downloaded `.tar.gz` files used in the locked study.

The loader verifies the digest before opening a Matrix Market member, checks the expected dimension and stored-entry count, symmetrizes the parsed matrix, and then constructs the registered affine numerical family.
