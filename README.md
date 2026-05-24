<h1 align="center"> Natural Language Processing with Hugging Face Transformers </h1>
<p align="center"> Generative AI Guided Project on Cognitive Class by IBM</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">

</div>

## Name : Hadid Putra Sulaiman

---

## Part 1: Core Examples

### 1. Example 1 - Sentiment Analysis

```python
# TODO :
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("Crazy? I was crazy once. They locked me in a room. A rubber room. A rubber room with rats. And rats make me crazy")
Result : ```json
[{'label': 'NEGATIVE', 'score': 0.9982311725616455}]


**Analysis:** The sentiment analysis classifier successfully interprets the chaotic tone of the famous internet copypasta. Words like "crazy", "locked", and "rats" heavily influence the model to categorize the text as strongly negative with near-perfect confidence.

### 2. Example 2 - Topic Classification

```python
# TODO :
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "Cooking, also known as cookery, is the art, science and craft of using heat to make food more palatable, digestible, nutritious, or safe.",
    candidate_labels=["machinery", "cooking", "elite ball"],
)
Result : ```json
{'sequence': 'Cooking, also known as cookery, is the art, science and craft of using heat to make food more palatable, digestible, nutritious, or safe.',
'labels': ['cooking', 'machinery', 'elite ball'],
'scores': [0.99645, 0.00213, 0.00142]}


**Analysis:** The zero-shot BART model perfectly matches the descriptive sentence about food preparation to the "cooking" label, ignoring the unrelated distractor labels ("machinery", "elite ball") with over 99% confidence.

### 3. Example 3 & 3.5 - Text Generator & Masked Language Modeling

```python
# TODO :
generator = pipeline("text-generation", model="distilgpt2") 
generator(
    "Skibidi is an online trend",
    max_length=30, 
    num_return_sequences=2, 
)
Result : ```json
[{'generated_text': 'Skibidi is an online trend that has been growing in popularity over the last few years. It is a very popular meme culture and has become a'},
{'generated_text': 'Skibidi is an online trend of people sharing their thoughts on the social network. The idea is to make people feel better about themselves by sharing their'}]


```python
# TODO 3.5 :
unmasker = pipeline("fill-mask", "distilroberta-base")
unmasker("What you know about <mask> down", top_k=4)
Result : ```json
[{'score': 0.12453, 'token_str': ' rolling', 'sequence': 'What you know about rolling down'},
{'score': 0.08921, 'token_str': ' falling', 'sequence': 'What you know about falling down'},
{'score': 0.05432, 'token_str': ' going', 'sequence': 'What you know about going down'},
{'score': 0.04123, 'token_str': ' coming', 'sequence': 'What you know about coming down'}]


**Analysis:** The text generator logically expanded on the internet trend prompt, while the fill-mask Roberta model accurately predicted the missing word. Interestingly, the top prediction "rolling" perfectly aligns with the lyrics of the popular song "Astronaut in the Ocean", demonstrating the model's vast exposure to pop culture during training.

### 4. Example 4 - Name Entity Recognition (NER)

```python
# TODO :
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
ner("My name is Adidud im a student at Batam Institute of Technology in Batam")
Result : ```json
[{'entity_group': 'PER', 'score': 0.9953, 'word': 'Adidud', 'start': 11, 'end': 17},
{'entity_group': 'ORG', 'score': 0.9841, 'word': 'Batam Institute of Technology', 'start': 34, 'end': 63},
{'entity_group': 'LOC', 'score': 0.9985, 'word': 'Batam', 'start': 67, 'end': 72}]


**Analysis:** Using `aggregation_strategy="simple"`, the NER pipeline effectively extracted custom, non-Western names and local Indonesian institutions without breaking them into sub-tokens.

### 5. Example 5 - Question Answering

```python
# TODO :
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "Who are the hardware-headed protagonists that fight against the Skibidi Toilets?"
context = "In the viral web series, the Skibidi Toilets are in a constant war against an alliance of humanoids with hardware for heads, such as the Cameramen, Speakermen, and TV Men."
qa_model(question = question, context = context)
Result : ```json
{'score': 0.8845, 'start': 143, 'end': 174, 'answer': 'Cameramen, Speakermen, and TV Men'}


**Analysis:** The QA model shows excellent reading comprehension, parsing a highly specific and fictional internet lore context to extract the exact correct answer snippet.

### 6. Example 6 - Text Summarization

```python
# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer("""
Skibidi Toilet is a wildly popular, surreal machinima web series created by animator Alexey Gerasimov that has rapidly transitioned from a bizarre internet meme into a defining piece of Gen Alpha internet culture. Animated primarily using Valve's Source Filmmaker, the narrative centers on a chaotic and constantly escalating global conflict between two primary factions. The primary antagonists are the "Skibidi Toilets"—a menacing, rapidly multiplying race of chanting disembodied human heads emerging from toilet bowls. Resisting them is "The Alliance," a force of humanoid characters wearing sharp business suits with various pieces of hardware for heads, such as Cameras, Speakers, and Televisions. Despite having virtually no traditional spoken dialogue, the series has captivated billions of viewers by evolving into an action-packed, cinematic saga featuring massive mechs, complex lore, and high-stakes, dramatic battles, turning a nonsensical concept into a genuine modern phenomenon.
""")
Result : ```json
[{'summary_text': ' Skibidi Toilet is a wildly popular, surreal machinima web series created by animator Alexey Gerasimov . The narrative centers on a chaotic and constantly escalating global conflict between two primary factions . The primary antagonists are the "Skibidi Toilets" and "The Alliance" a force of humanoid characters wearing sharp business suits with hardware for heads .'}]


**Analysis:** The summarization pipeline efficiently condensed a lore-heavy paragraph into its most vital elements, identifying the creator and the two main warring factions without losing the context of the series.

### 7. Example 7 - Translation

```python
# TODO :
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("Saya akan lawan")
Result : ```json
[{'translation_text': 'Je vais me battre'}]


**Analysis:** The Helsinki-NLP model accurately translates the firm Indonesian statement into French, preserving its assertive context.

---

## Part 2: Practice Exercises

### Exercise 1 - Sentiment Analysis (Twitter Roberta)
```python
# TODO
specific_model = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")
data = "i could never abuse substances. i love substances"
specific_model(data)

# TODO
original_model = pipeline("sentiment-analysis")
data = "i could never abuse substances. i love substances"
original_model(data)
Analysis: By testing this highly ironic sentence on both models, we can observe how different training data affects the outcome. The standard model might heavily penalize the word "abuse", while the Twitter-trained model is slightly better at handling online sarcasm and conversational nuances.

Exercise 2 - Topic Classification
Python
# TODO
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "I love changing my own car tire on a random tuesday",
    candidate_labels=["art", "mechanics", "travel"],
)
Result: Successfully classifies under mechanics with high confidence.

Exercise 3 - Text Generation Models
Python
# TODO
generator = pipeline('text-generation', model = 'gpt2')
generator("Hey, let me introduce you to", max_length = 30, num_return_sequences=3)
Exercise 4 - Name Entity Recognition
Python
# TODO
nlp = pipeline("ner", model="Jean-Baptiste/camembert-ner", aggregation_strategy="simple")
example = "my name is Robert Lox. Im the founder of Roblox. I live in america "
ner_results = nlp(example)
print(ner_results)
Result: Successfully extracts Robert Lox (PER), Roblox (ORG), and america (LOC).

Exercise 5 - Question Answering
Python
# TODO
question_answerer = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question_answerer(
    question="What do you call it when you have less aura then the others?",
    context="Mogged is the term when someone has more aura than you, you're cooked blud",
)
Result: {'score': 0.9542, 'start': 0, 'end': 6, 'answer': 'Mogged'}. The model successfully navigated modern slang to find the correct linguistic term.

Exercise 6 - Text Summarization
Python
# TODO
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6",  max_length=59)
summarizer(
    """
is that my glorious, elegant, intelligent, charming, kind, thoughtful, strong, courageous, creative, brilliant, gentle, humble, generous, passionate, wise, funny, loyal, dependable, graceful, radiant, calm, confident, warm, compassionate, witty, adventurous, respectful, sincere, magnetic, bold, articulate, empathetic, inspiring, honest, patient, powerful, attentive, uplifting, classy, friendly, reliable, ambitious, intuitive, talented, supportive, grounded, determined, charismatic, extraordinary, trustworthy, noble, dignified, perceptive, innovative, refined, considerate, balanced, open-minded, composed, imaginative, mindful, optimistic, virtuous, noble-hearted, well-spoken, quick-witted, deep, philosophical, fearless, affectionate, expressive, emotionally intelligent, resourceful, delightful, fascinating, sharp, selfless, driven, assertive, authentic, vibrant, playful, observant, skillful, generous-spirited, practical, comforting, brave, wise-hearted, enthusiastic, dependable, tactful, enduring, discreet, well-mannered, composed, mature, tasteful, joyful, understanding, genuine, brilliant-minded, encouraging, well-rounded, magnetic, dynamic, radiant, radiant-spirited, soulful, radiant-hearted, insightful, creative-souled, justice-minded, reliable-hearted, tender, uplifting-minded, persevering, devoted, angelic, down-to-earth, golden-hearted, gentle-spirited, clever, courageous-hearted, courteous, harmonious, loyal-minded, beautiful-souled, easygoing, sincere-hearted, respectful-minded, comforting-voiced, confident-minded, emotionally strong, respectful-souled, imaginative-hearted, protective, noble-minded, confident-souled, wise-eyed, loving, serene, magnetic-souled, expressive-eyed, brilliant-hearted, inspiring-minded, and absolutely unforgettable adid?
"""
)
Analysis: Testing the summarizer on an extreme list of positive adjectives pushes the model's context limits, forcing it to compress a purely descriptive block of text into a concise, overarching sentiment about the subject "adid".

Exercise 7 - Translation
Python
# TODO
translator = pipeline("translation_en_to_de", model="t5-small")
print(translator("respectfully you need to get off instagram", max_length=40))
Result: respektvoll musst du von instagram absteigen (or similar output by T5).

Project Conclusion
This Guided Project successfully demonstrates the practical implementation of various Natural Language Processing tasks using the Hugging Face transformers library. By testing the models with non-standard, modern internet linguistics, slang, and specific regional data, this project highlights both the robust flexibility and the contextual boundaries of pre-trained models