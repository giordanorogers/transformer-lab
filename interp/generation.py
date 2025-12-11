from nnsight import LanguageModel

def llama_generate(
    model: LanguageModel,
    prompt: str,
    output: Literal[
        'next_token',
        'last_token_logits',
        'attention_pattern'
    ]
):
    if output in ['next_token', 'last_token_logits']
        with model.trace(prompt):
            last_token_logits = model.output[0][0, -1, :].save()
        if output == 'last_token_logits':
            return last_token_logits
        elif output == 'next_token':
            probs = torch.softmax(last_token_logits, dim=0)
            token_id = torch.argmax(probs)
            return model.tokenizer.decode(token_id)
     elif output == 'attention_pattern':
        with model.trace(prompt, output_attentions=True):
            attn_pattern = model.output.attentions[-1].save()
        