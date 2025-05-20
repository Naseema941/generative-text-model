from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def generate_text(prompt, max_length=150, temperature=0.7, top_p=0.9):
    """
    Generate text from a given prompt using GPT-2.

    Args:
        prompt (str): Starting prompt.
        max_length (int): Maximum length of generated sequence.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.

    Returns:
        str: Generated text.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=0,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    while True:
        prompt = input("\nEnter a prompt (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        generated = generate_text(prompt)
        print("\nGenerated Paragraph:\n")
        print(generated)
