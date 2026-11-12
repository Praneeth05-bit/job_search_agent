"""Model-layer errors, independent of any provider SDK.

Every failure from a model provider is converted into one of these. Each
class says whether retrying could help, which the retry helper uses to
decide what to do, and carries a short, friendly ``user_message`` that the
API and UI can show to a student.

    retryable:      temporary problems that may succeed on a later attempt
    not retryable:  permanent problems where retrying only wastes quota

The exception text (``str(error)``) is the technical detail for logs and
developers; ``user_message`` is what end users see. It never includes
technical details or configuration hints.
"""

from typing import ClassVar


class ModelError(Exception):
    """Base class for every model failure. Catch this to handle them all."""

    retryable: ClassVar[bool] = False
    user_message: ClassVar[str] = (
        "Something went wrong with the AI model. Please try again."
    )


class ModelTimeoutError(ModelError):
    """The model did not answer within the configured timeout."""

    retryable = True
    user_message = "The AI model took too long to respond. Please try again."


class ModelRateLimitError(ModelError):
    """The provider rejected the call for too many requests (HTTP 429)."""

    retryable = True
    user_message = (
        "The AI model is receiving too many requests right now. "
        "Please try again in a minute."
    )


class ModelUnavailableError(ModelError):
    """The provider had a temporary server problem (HTTP 5xx)."""

    retryable = True
    user_message = "The AI model is busy right now. Please try again in a few minutes."


class ModelAuthError(ModelError):
    """The API key is missing, invalid, or not allowed (HTTP 401 or 403)."""

    user_message = (
        "The AI service is not configured correctly. Please contact the site owner."
    )


class ModelResponseError(ModelError):
    """The model answered, but the answer was unusable or invalid."""

    user_message = (
        "The AI model returned an answer we could not use. "
        "Please try again or rephrase your request."
    )
