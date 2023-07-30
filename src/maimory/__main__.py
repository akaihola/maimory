import os
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from textwrap import dedent, indent
from typing import Union

import openai
import pypandoc
from maimory.tokens import num_tokens_from_messages


def main():
    parser = ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    history = load_conversation(args.path)
    assert_ends_with_prompt(history)
    conversation = fit_into_tokens(history)
    response = interact(conversation)
    append_completion(history, response)
    save_conversation(args.path, history)
    conversation.append(history[-1])
    print_conversation(conversation)


@dataclass
class Utterance:
    index: int
    timestamp: datetime
    role: str
    notes: str
    content: str

    def to_string(self):
        indented_content = indent(self.content, "  ")
        iso_timestamp = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return (
            "### "
            f"{self.role} "
            f"{iso_timestamp} "
            f"[{self.index}]{self.notes}:\n\n"
            f"{indented_content}"
        )

    def to_dict(self):
        return {"role": self.role, "content": self.content}


def load_conversation(path: Path) -> list[Utterance]:
    content = path.read_text()
    utterances = parse_utterances(content)
    return utterances


def parse_utterances(content: str) -> list[Utterance]:
    non_indented_line_matches = re.finditer(r"(?:^|\n)[#a-z]", content)
    non_indented_line_indices: list[Union[int, None]] = [
        match.start() for match in non_indented_line_matches
    ]
    chunks = [
        content[i:j]
        for i, j in zip(
            non_indented_line_indices, non_indented_line_indices[1:] + [None]
        )
    ]
    utterances = [parse_utterance(chunk, index) for index, chunk in enumerate(chunks)]
    return utterances


def parse_utterance(chunk: str, index: int) -> Utterance:
    lines = chunk.strip().splitlines()
    metadata_match = re.match(
        r"""
        (?: \#* \s* )?
        (?P<role>\w+)
        (?:
          \s+
          (?P<date>\d{4}-\d{2}-\d{2}) \s+
          (?P<time>\d{2}:\d{2}:\d{2}) \s+
          \[\d+](?P<notes>.*?)
        )?
        (?: : \s*)?
        $
        """,
        lines[0],
        re.VERBOSE,
    )
    if metadata_match is None:
        raise ValueError(f"Could not parse metadata from {lines[0]!r}")
    metadata = metadata_match.groupdict()
    if metadata["date"]:
        timestamp = datetime.fromisoformat(f"{metadata['date']}T{metadata['time']}Z")
    else:
        timestamp = datetime.utcnow()
    content = dedent("\n".join(lines[1:]).strip("\n"))
    return Utterance(
        index, timestamp, metadata["role"], metadata["notes"] or "", content
    )


def assert_ends_with_prompt(history: list[Utterance]) -> None:
    if history[-1].role != "user":
        raise ValueError("Last utterance must be from user")


def fit_into_tokens(history: list[Utterance]) -> list[Utterance]:
    result = history[:1] + history[-1:]
    total_tokens = num_tokens_from_messages(utterance.to_dict() for utterance in result)
    for utterance in history[-2:0:-1]:
        tokens = num_tokens_from_messages([utterance.to_dict()])
        if total_tokens + tokens > 4096:
            break
        total_tokens += tokens
        result.insert(1, utterance)
    return result


@dataclass
class Choice:
    role: str
    content: str


def fill_paragraphs(content: str) -> str:
    Path("error.md").write_text(content)
    result = pypandoc.convert_text(
        content, "markdown", format="markdown", extra_args=["--columns=86"]
    )
    return result


def interact(conversation: list[Utterance]) -> Choice:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[utterance.to_dict() for utterance in conversation],
    )
    completion = response.choices[0].message
    completion.content = fill_paragraphs(completion.content)
    return completion


def append_completion(history: list[Utterance], response: Choice) -> None:
    history.append(
        Utterance(len(history), datetime.utcnow(), response.role, "", response.content)
    )


def save_conversation(path: Path, history: list[Utterance]) -> None:
    path.write_text("\n\n".join(utterance.to_string() for utterance in history))


def print_conversation(history: list[Utterance]) -> None:
    for utterance in history:
        print(utterance.to_string())
        print()


if __name__ == "__main__":
    main()
