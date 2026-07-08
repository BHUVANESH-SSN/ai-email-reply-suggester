import argparse
import json

from generator.config import GEN_MODEL, create_chat_completion
from generator.prompt_templates import build_messages
from generator.retrieve import load_dataset, retrieve_similar


def generate_reply(subject: str, body: str, dataset_rows: list, k: int = 3) -> dict:
    retrieved = retrieve_similar(subject, body, dataset_rows, k=k)
    messages = build_messages(subject, body, retrieved)
    response = create_chat_completion(
        model=GEN_MODEL,
        messages=messages,
        temperature=0.7,
    )
    reply_text = response.choices[0].message.content.strip()
    return {
        "reply": reply_text,
        "retrieved": [
            {"id": item["row"]["id"], "similarity": item["similarity"]} for item in retrieved
        ],
        "model": GEN_MODEL,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--dataset", default="dataset/emails.jsonl")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    dataset_rows = load_dataset(args.dataset)
    result = generate_reply(args.subject, args.body, dataset_rows, k=args.k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
