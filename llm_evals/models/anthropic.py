import os

from anthropic import Anthropic

from llm_evals.models.base import LLMInterface


class AnthropicModel(LLMInterface):

    @property
    def _client(self):
        return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def call(self, prompt, system_prompt, **kwargs):
        return (
            self.client.messages.create(  # type: ignore
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                **(kwargs | self.hyperparameters),
            )
            .content[0]
            .text
        )
