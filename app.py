# app.py
import gradio as gr

# Dummy translator for demo purposes
def translate_english_to_tamil(text):
    # Replace with your model's inference logic later
    sample_translations = {
        "hello": "வணக்கம்",
        "how are you?": "நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "thank you": "நன்றி"
    }
    return sample_translations.get(text.lower(), "மொழிபெயர்ப்பு இல்லை (Translation not available)")

demo = gr.Interface(
    fn=translate_english_to_tamil,
    inputs=gr.Textbox(lines=2, placeholder="Enter English sentence..."),
    outputs=gr.Textbox(label="Tamil Translation"),
    title="English ➜ Tamil Translator",
    description="This is a demo for testing Transformer model inference in Hugging Face Spaces."
)

if __name__ == "__main__":
    demo.launch()
