"""**Embeddings** interface."""

from abc import ABC, abstractmethod
from enum import Enum

from langchain_core.runnables.config import run_in_executor


class ContentType(Enum):
    text = "text"
    image_url = "image_url"
    image_base64 = "image_base64"


class Content(ABC):
    type: ContentType
    data: str


class MultimodalInput(ABC):
    input: list[Content]


class MultimodalEmbeddings(ABC):
    """Interface for multimodal embedding models.

    This is an interface meant for implementing multimodal embedding models.

    Multimodal embedding models are used to map text and media (image, video, etc.) to a
    vector (a point in n-dimensional space).

    Texts or media that are similar will usually be mapped to points that are close to
    each other in this space. The exact details of what's considered "similar" and how
    "distance" is measured in this space are dependent on the specific embedding model.

    This abstraction contains a method for embedding a list of documents and a method
    for embedding a query text. The embedding of a query text is expected to be a single
    vector, while the embedding of a list of inputs is expected to be a list of
    vectors.

    Usually the query embedding is identical to the document embedding, but the
    abstraction allows treating them independently.

    In addition to the synchronous methods, this interface also provides asynchronous
    versions of the methods.

    By default, the asynchronous methods are implemented using the synchronous methods;
    however, implementations may choose to override the asynchronous methods with
    an async native implementation for performance reasons.
    """

    @abstractmethod
    def embed_documents(
            self,
            multimodal_inputs: list[MultimodalInput]
    ) -> list[list[float]]:
        """Embed search docs.

        Args:
            multimodal_inputs: List of contents to embed.

        Returns:
            List of embeddings.
        """

    @abstractmethod
    def embed_query(self, multimodal_input: MultimodalInput) -> list[float]:
        """Embed query text.

        Args:
            multimodal_input: Content to embed.

        Returns:
            Embedding.
        """

    async def aembed_documents(
            self,
            multimodal_inputs: list[MultimodalInput]
    ) -> list[list[float]]:
        """Asynchronous Embed search docs.

        Args:
            multimodal_inputs: List of contents to embed.

        Returns:
            List of embeddings.
        """
        return await run_in_executor(None, self.embed_documents, multimodal_inputs)

    async def aembed_query(self, multimodal_input: MultimodalInput) -> list[float]:
        """Asynchronous Embed query text.

        Args:
            multimodal_input: Content to embed.

        Returns:
            Embedding.
        """
        return await run_in_executor(None, self.embed_query, multimodal_input)
