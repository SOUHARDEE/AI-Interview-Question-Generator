from retrieval import search
from generator import generate_questions, generate_general_questions


def rag_system(query, skill=None, difficulty="Medium"):

    retrieved, distances = search(query, skill=skill, difficulty=difficulty, k=3)

    if not retrieved:
        return generate_general_questions(query, difficulty)

    # Confidence threshold
    if distances is None or distances[0] > 1.2:
        return generate_general_questions(query, difficulty)

    context = "\n".join([f"- {q}" for q in retrieved])

    return generate_questions(context, difficulty)


if __name__ == "__main__":
    topic = input("Enter topic: ")
    difficulty = input("Enter difficulty (Easy/Medium/Hard): ")

    result = rag_system(
        query=topic,
        skill=None,
        difficulty=difficulty
    )

    print("\nGenerated Questions:\n")
    print(result)