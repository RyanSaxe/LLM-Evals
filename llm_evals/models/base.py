import abc
from functools import cached_property

from dotenv import load_dotenv

load_dotenv()


class LLMInterface(abc.ABC):

    def __init__(self, model: str, **hyperparameters):
        self.model = model
        self.hyperparameters = hyperparameters
        # TODO: implement memory -- or just leverage LangChain instead of my own interface
        # self._message_memory = []

    @cached_property
    def client(self):
        return self._client

    @abc.abstractproperty
    def _client(self): ...

    @abc.abstractmethod
    def __call__(self, prompt: str, system_prompt: str, **kwargs) -> str: ...
