import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading TinyLlama model...")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("Model loaded successfully!\n")


def clean_output(text):
    lines = text.strip().split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line[0].isdigit():
            if "." in line:
                line = line.split(".", 1)[-1].strip()
            elif ")" in line:
                line = line.split(")", 1)[-1].strip()

        if line.startswith("-"):
            line = line[1:].strip()

        if "?" in line and len(line) > 20:
            question = line.split("?")[0] + "?"
            cleaned.append(f"- {question}")

    return cleaned[:3]


def run_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.5,
            top_p=0.9,
            do_sample=True
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = decoded[len(prompt):].strip()
    return generated


def generate_questions(context, difficulty="Medium"):
    prompt = f"""<|system|>
You are an expert technical interviewer.
<|user|>
Existing interview questions:
{context}

Generate exactly 3 NEW {difficulty} level interview questions.
Output only questions ending with '?'.
Do not explain anything.
<|assistant|>
"""

    response = run_model(prompt)
    questions = clean_output(response)

    if len(questions) == 0:
        questions = [
            "- What is a key concept related to this topic?",
            "- How is this concept applied in real-world systems?",
            "- What challenges are associated with this concept?"
        ]

    return "\n".join(questions)


def generate_general_questions(topic, difficulty="Medium"):
    prompt = f"""<|system|>
You are an expert technical interviewer.
<|user|>
Generate exactly 3 {difficulty} level interview questions about {topic}.
Output only questions ending with '?'.
Do not explain anything.
<|assistant|>
"""

    response = run_model(prompt)
    questions = clean_output(response)

    if len(questions) == 0:
        questions = [
            f"- What is a fundamental concept of {topic}?",
            f"- How is {topic} applied in real-world systems?",
            f"- What challenges are associated with {topic}?"
        ]

    return "\n".join(questions)


if __name__ == "__main__":
    print(generate_general_questions("Deep Learning", "Medium"))