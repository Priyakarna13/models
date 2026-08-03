class ModelRegistryError(Exception):
    """Base exception for expected user-facing failures."""


class ValidationError(ModelRegistryError):
    """A model definition does not satisfy its schema or repository policy."""


class SourcePolicyError(ModelRegistryError):
    """A source URI violates the immutable-source policy."""


class FetchError(ModelRegistryError):
    """A source could not be downloaded or verified."""
