"""User frustration evaluation for the travel agent's traces in Phoenix.

What it does:
1. Pulls the travel-agent spans from the local Phoenix server.
2. Keeps the root span of each trace (one per user request), which holds the
   user's message as input and the agent's final state as output.
3. Asks an LLM judge to label each one "frustrated" or "not_frustrated",
   with a short explanation.
4. Logs the results back to Phoenix as span annotations named
   "user_frustration", so they show next to each trace and can be filtered.

Run it while Phoenix is running:
    poetry run python evaluate_frustration.py
"""

import os

from dotenv import load_dotenv
from phoenix.client import Client
from phoenix.evals import LLM, bind_evaluator, create_classifier, evaluate_dataframe
from phoenix.evals.utils import to_annotation_dataframe

load_dotenv()

PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "travel-agent")
EVAL_NAME = "user_frustration"
JUDGE_MODEL = "gpt-4o"

FRUSTRATION_TEMPLATE = """You are evaluating one exchange between a user and a travel assistant.

Decide whether the user is frustrated.

Signs of frustration include:
- saying a previous answer was wrong, unhelpful or ignored what they asked
- repeating or correcting a request they already made
- impatience, sarcasm, insults or angry language (for example all caps or many exclamation marks)

Frustration can range from mild to extreme. A user who simply asks a question,
even a short or blunt one, is not frustrated. Base your decision on the user's
message; use the assistant's output only as context.

[User input]: {input}

[Assistant output]: {output}

Answer "frustrated" or "not_frustrated"."""


def main() -> None:
    client = Client()  # connects to http://localhost:6006 by default

    # 1. Pull spans for the project
    spans_df = client.spans.get_spans_dataframe(project_identifier=PROJECT_NAME)
    if spans_df.empty:
        print(f"No spans found in project '{PROJECT_NAME}'. Send some requests first.")
        return

    # 2. Keep only root spans (one per user request / trace)
    root_spans = spans_df[spans_df["parent_id"].isna()].copy()
    if "context.span_id" in root_spans.columns:
        root_spans = root_spans.set_index("context.span_id", drop=False)
    print(f"Evaluating {len(root_spans)} traces from '{PROJECT_NAME}'...")

    # 3. Build the LLM-as-judge classifier and map it to the span columns
    judge = LLM(provider="openai", model=JUDGE_MODEL)
    frustration_eval = create_classifier(
        name=EVAL_NAME,
        prompt_template=FRUSTRATION_TEMPLATE,
        llm=judge,
        choices={"frustrated": 1.0, "not_frustrated": 0.0},
        direction="minimize",  # fewer frustrated users is better
    )
    bound_eval = bind_evaluator(
        evaluator=frustration_eval,
        input_mapping={
            "input": "attributes.input.value",
            "output": "attributes.output.value",
        },
    )

    results_df = evaluate_dataframe(dataframe=root_spans, evaluators=[bound_eval])

    # 4. Log the results back to Phoenix as span annotations
    annotations = to_annotation_dataframe(dataframe=results_df)
    client.spans.log_span_annotations_dataframe(dataframe=annotations)

    print("\nResults:")
    if "label" in annotations.columns:
        print(annotations["label"].value_counts().to_string())
    print(f"\nLogged {len(annotations)} '{EVAL_NAME}' annotations to Phoenix.")


if __name__ == "__main__":
    main()
