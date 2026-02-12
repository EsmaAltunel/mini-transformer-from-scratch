import torch
import json
import gradio as gr
from model import Model 
from tokenizer import Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_all():
    checkpoint = torch.load("model.pth", map_location=device)
    config = checkpoint['config']
    
    model = Model(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        context_length=config['context_length'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with open("temp_vocab.json", "w") as f:
        json.dump(checkpoint['vocab'], f)
    
    tokenizer = Tokenizer("temp_vocab.json")
    return model, tokenizer

model, tokenizer = load_all()

def swiftie_ai(prompt, tokens, temp, top_k):
    input_ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=int(tokens), 
            temperature=float(temp), 
            top_k=int(top_k)
        )
    return tokenizer.decode(output_ids[0])

demo = gr.Interface(
    fn=swiftie_ai,
    inputs=[
        gr.Textbox(label="Bir dize başla:", placeholder="i remember it..."),
        gr.Slider(10, 100, value=40, label="Kelime Sayısı"),
        gr.Slider(0.1, 1.5, value=0.8, label="Yaratıcılık (Temperature)"),
        gr.Slider(1, 100, value=30, label="Kelime Havuzu (Top-K)")
    ],
    outputs=gr.Textbox(label="AI Taylor:"),
    title="🎸 Swiftie AI 🏎️",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(share = False)