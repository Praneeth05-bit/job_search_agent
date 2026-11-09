"""Model-layer errors, independent of any provider SDK.

Every failure from a model provider is converted into one of these. Each
class says whether retrying could help, which the retry helper uses to
decide what to do.

    retryable:      temporary problems that may succeed on a later attempt
    not retryable:  permanent problems where retrying only wastes quota
"""

from typing import ClassVar


class ModelError(Exception):
    """Base class for every model failure. Catch this to handle them all."""

    retryable: ClassVar[bool] = False


class ModelTimeoutError(ModelError):
    """The model did not answer within the configured timeout."""

    retryable = True


class ModelRateLimitError(ModelError):
    """The provider rejected the call for too many requests (HTTP 429)."""

    retryable = True


class ModelUnavailableError(ModelError):
    """The provider had a temporary server problem (HTTP 5xx)."""

    retryable = True


class ModelAuthError(ModelError):
    """The API key is missing, invalid, or not allowed (HTTP 401 or 403)."""


class ModelResponseError(ModelError):
    """The model answered, but the answer was unusable or invalid."""
