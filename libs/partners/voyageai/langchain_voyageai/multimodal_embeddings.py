import logging
from typing import Any, List, Optional

import voyageai  # type: ignore
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    model_validator,
)
from typing_extensions import Self

from langchain_core.embeddings import MultimodalEmbeddings, MultimodalInput, Content
from langchain_core.utils import secret_from_env

logger = logging.getLogger(__name__)


class VoyageAIMultimodalEmbeddings(BaseModel, MultimodalEmbeddings):
    """VoyageAIMultimodalEmbeddings embedding model.

    Example:
        .. code-block:: python

            from langchain_voyageai import VoyageAIMultimodalEmbeddings

            model = VoyageAIMultimodalEmbeddings()
    """

    _client: voyageai.Client = PrivateAttr()
    _aclient: voyageai.client_async.AsyncClient = PrivateAttr()
    model: str
    show_progress_bar: bool = False
    truncation: Optional[bool] = None
    voyage_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env(
            "VOYAGE_API_KEY",
            error_message="Must set `VOYAGE_API_KEY` environment variable or "
            "pass `api_key` to VoyageAIEmbeddings constructor.",
        ),
    )

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    @classmethod
    def _langchain_content_to_content(cls, content: Content) -> dict:
        result = {
            "type": content.type
        }
        if content.text is not None:
            result["text"] = content.text
        if content.image_url is not None:
            result["image_url"] = content.image_url
        if content.image_base64 is not None:
            result["image_base64"] = content.image_base64

        return result

    @classmethod
    def _langchain_multimodal_input_to_input(cls, multimodal_input: MultimodalInput) -> dict:
        return {
            "content": [cls._langchain_content_to_content(x) for x in multimodal_input.input]
        }

    @model_validator(mode="before")
    @classmethod
    def default_values(cls, values: dict) -> Any:
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that VoyageAI credentials exist in environment."""
        api_key_str = self.voyage_api_key.get_secret_value()
        self._client = voyageai.Client(api_key=api_key_str)
        self._aclient = voyageai.client_async.AsyncClient(api_key=api_key_str)
        return self

    def embed_documents(self, multimodal_inputs: list[MultimodalInput]) -> List[List[float]]:
        """Embed search docs."""

        return self._client.multimodal_embed(
            [self._langchain_multimodal_input_to_input(x) for x in multimodal_inputs],
            model=self.model,
            input_type="document",
            truncation=self.truncation,
        ).embeddings

    def embed_query(self, multimodal_input: MultimodalInput) -> List[float]:
        """Embed query text."""
        return self._client.multimodal_embed(
            [self._langchain_multimodal_input_to_input(multimodal_input)],
            model=self.model,
            input_type="query",
            truncation=self.truncation
        ).embeddings[0]

    async def aembed_documents(self, multimodal_inputs: list[MultimodalInput]) -> List[List[float]]:
        result = await self._aclient.multimodal_embed(
            [self._langchain_multimodal_input_to_input(x) for x in multimodal_inputs],
            model=self.model,
            input_type="document",
            truncation=self.truncation,
        )
        return result.embeddings

    async def aembed_query(self, multimodal_input: MultimodalInput) -> List[float]:
        result = await self._aclient.multimodal_embed(
            [self._langchain_multimodal_input_to_input(multimodal_input)],
            model=self.model,
            input_type="document",
            truncation=self.truncation,
        )
        return result.embeddings[0]
