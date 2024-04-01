import os

from anthropic import Anthropic

from llm_evals.models.base import LLMInterface


class OpenAIModel(LLMInterface):

    @property
    def _client(self):
        return Anthropic(api_key=os.environ.get("OPENAI_API_KEY"))

    def __call__(self, system_prompt, prompt, **kwargs):
        return (
            self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                **(kwargs | self.hyperparameters),
            )
            .content[0]
            .text
        )
