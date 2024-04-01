import abc
import logging
import time
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
    def call(self, prompt: str, system_prompt: str, **kwargs) -> str: ...

    def __call__(self, prompt: str, system_prompt: str, _n_retries: int = 5, _sleep: float = 20, **kwargs) -> str:
        try:
            return self.call(prompt, system_prompt, **kwargs)
        except Exception as e:
            if _n_retries <= 0:
                logging.debug("failed to fet a valid response")
                return ""
            logging.debug(e)
            logging.info(f"{_n_retries} retries left")
            time.sleep(_sleep)
            return self(prompt, system_prompt, _n_retries=_n_retries - 1, _sleep=_sleep, **kwargs)
