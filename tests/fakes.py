import json
from types import SimpleNamespace


class FakeClient:
    """Stands in for OpenAI: each create() call replays one scripted response."""

    def __init__(self, responses):
        self.responses = self
        self._scripted = iter(responses)
        self.requests = []  # what we sent, so tests can inspect tools and instructions

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return iter(next(self._scripted))

    @property
    def calls(self):
        return len(self.requests)


def text_response(text):
    done = SimpleNamespace(type="message")
    return [
        SimpleNamespace(type="response.output_text.delta", delta=text),
        SimpleNamespace(type="response.completed", response=SimpleNamespace(output=[done])),
    ]


def tool_response(name, arguments):
    item = SimpleNamespace(
        type="function_call", name=name, arguments=json.dumps(arguments), call_id="call_1"
    )
    return [SimpleNamespace(type="response.completed", response=SimpleNamespace(output=[item]))]
