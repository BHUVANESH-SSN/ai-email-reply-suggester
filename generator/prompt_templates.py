SYSTEM_PROMPT = """You are an assistant that drafts suggested email replies on behalf of a person handling their inbox.

Rules:
- Address every distinct question or request in the incoming email.
- Match the tone shown in the example replies below (formal/casual/apologetic).
- Include a greeting and a sign-off, consistent with the examples.
- Do NOT invent facts, dates, prices, order numbers, or commitments that are not present in the incoming email or in the example replies below.
- Keep length roughly consistent with the example replies for this category.
- Output ONLY the reply text. No preamble, no explanation, no subject line."""


def build_few_shot_block(retrieved: list) -> str:
    blocks = []
    for item in retrieved:
        row = item["row"]
        blocks.append(
            "Example incoming email:\n"
            f"Subject: {row['incoming_email']['subject']}\n"
            f"{row['incoming_email']['body']}\n\n"
            "Example reply that was actually sent:\n"
            f"{row['sent_reply']}"
        )
    return "\n\n---\n\n".join(blocks)


def build_messages(subject: str, body: str, retrieved: list) -> list:
    few_shot = build_few_shot_block(retrieved)
    user_content = (
        f"Here are {len(retrieved)} similar past emails and the replies that were sent:\n\n"
        f"{few_shot}\n\n---\n\n"
        "Now write a suggested reply to this new incoming email:\n\n"
        f"Subject: {subject}\n{body}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
