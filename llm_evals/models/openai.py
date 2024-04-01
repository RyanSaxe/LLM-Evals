import os

from openai import OpenAI

from llm_evals.models.base import LLMInterface


class OpenAIModel(LLMInterface):

    @property
    def _client(self):
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def call(self, prompt, system_prompt, **kwargs):
        return (
            self.client.chat.completions.create(  # type: ignore
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                **(kwargs | self.hyperparameters),
            )
            .choices[0]
            .message.content
        )
