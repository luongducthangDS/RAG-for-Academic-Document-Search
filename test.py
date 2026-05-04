from __future__ import annotations

import os

from google import genai


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY or GOOGLE_API_KEY.")

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents="Ban la gi?",
        )
        print(response.text or "[empty response]")
    except Exception as exc:
        print(f"Loi: {exc}")


if __name__ == "__main__":
    main()
