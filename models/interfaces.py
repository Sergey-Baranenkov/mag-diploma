import typing
from abc import ABC, abstractmethod


class QueryEncoder(ABC):
    @abstractmethod
    def get_query_embed(
            self,
            modality: typing.Literal['text', 'audio', 'hybrid'],
            audio=None,
            text=None,
            use_text_ratio: float = 1,
            device=None
    ):  # todo typing
        pass


class SSModel(ABC):  # todo typing
    @abstractmethod
    def __call__(self, dict):
        pass
