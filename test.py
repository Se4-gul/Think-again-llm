import anthropic

client = anthropic.Anthropic(api_key = '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')

response = client.messages.create(
    model = 'claude-haiku-4-5-20251001',
    max_tokens = 100,
    messages = [
        {"role": "user", "content": "Say hello"}
    ]
)

print(response.content[0].text)