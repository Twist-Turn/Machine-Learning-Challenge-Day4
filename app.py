from flask import Flask, render_template, request
from transformers import BartForConditionalGeneration, BartTokenizer

app = Flask(__name__)

# Load pre-trained BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_text():
    if request.method == 'POST':
        text = request.form['text']
        # Tokenize input text
        inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)

        # Generate summary
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)

        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return render_template('index.html', original_text=text, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
