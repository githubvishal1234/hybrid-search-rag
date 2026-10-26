"""Prompt for grounded, cited answers.

Sources are numbered [1]..[n] and the model cites by number. Asking a model to reproduce
"[sebi_master_circular_lodr_2026.pdf, p.146]" invites typos; a number is easy to emit and easy to
check, and answer.py maps it back to file + page and drops any number that wasn't in the context.
"""

REFUSAL = "I couldn't find this in the documents."

SYSTEM = f"""You answer questions about Indian financial regulations and the Union Budget using \
ONLY the numbered sources provided.

Rules:
- Use only facts stated in the sources. Do not use outside knowledge.
- Put the source number in square brackets after every sentence that uses it, e.g. [2] or [1][3].
- Quote figures, section numbers and dates exactly as written in the sources.
- If the sources do not contain the answer, reply exactly: {REFUSAL}
- Be concise: a few sentences or a short list."""


def format_sources(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        pages = f"p.{c['page_start']}" + (
            f"-{c['page_end']}" if c["page_end"] != c["page_start"] else ""
        )
        heading = f" | {c['section_heading']}" if c["section_heading"] else ""
        parts.append(f"[{i}] {c['title']} ({pages}){heading}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def user_prompt(question: str, chunks: list[dict]) -> str:
    return f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"
