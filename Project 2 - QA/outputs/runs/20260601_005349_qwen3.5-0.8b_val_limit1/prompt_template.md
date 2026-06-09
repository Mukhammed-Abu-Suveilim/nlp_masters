# Prompt template

Placeholders: `{question}` = full question string, `{excerpt}` = retrieved contract text.

```
Based ONLY on the contract text below, answer with the shortest possible exact quote from the text.

QUESTION: {question}

CONTRACT TEXT:
{excerpt}

RULES:
- Return ONLY exact words/phrases copied from the contract
- One short phrase or sentence maximum when possible
- If not in the text, reply exactly: NOT FOUND

ANSWER:
```
