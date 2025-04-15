from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Modellname auf Hugging Face
model_name = "phonemetransformers/GPT2-85M-CHAR-TXT"

# Tokenizer und Modell laden
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Eingabetext
input_text = "test senctence"

# Tokenisieren des Texts mit padding, truncation und max_length
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=30)

# Setzen von pad_token_id, falls es nicht automatisch gesetzt wird
model.config.pad_token_id = tokenizer.eos_token_id

# Modell-Antwort generieren mit max_length (auch hier festgelegt)
outputs = model.generate(
    inputs['input_ids'], 
    max_length=50,            # Maximale Länge für den generierten Text
    num_return_sequences=1,   # Eine Antwort zurückgeben
    attention_mask=inputs['attention_mask'],
    do_sample=True,           # Optional, für zufällige Generation
    top_k=50,                 # Optional, für Begrenzung der Auswahl bei der nächsten Tokenwahl
    top_p=0.95                # Optional, für nucleus sampling (Kernbereich der Auswahl)
)

# Decodieren und Ausgabe anzeigen
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
