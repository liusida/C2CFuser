# =============================================================
# Cache-to-Cache (C2C) Benchmarking Script
# =============================================================

import torch
from datasets import load_dataset
from tqdm import tqdm
from utils import (
    load_models_and_tokenizer,
    load_checkpoint,
    MMLU_REDUX_CATEGORIES,
    get_chat_template,
    get_mmlu_question,
    check_answer,
)


# -------------------------------------------------------------
# Benchmarking Function
# -------------------------------------------------------------
def benchmark(
    checkpoint_step,
    ckpt_dir="checkpoints",
    categories=None,
    max_questions_per_category=None,
    max_new_tokens=64,
    verbose=True,
):
    """
    Benchmark a checkpoint on MMLU Redux dataset.

    Args:
        checkpoint_step: Step number of checkpoint to evaluate
        ckpt_dir: Directory containing checkpoints
        categories: List of categories to evaluate on (None = all categories)
        max_questions_per_category: Max questions per category (None = all)
        max_new_tokens: Maximum tokens to generate
        verbose: Print progress and results

    Returns:
        dict: Results containing accuracy per category and global accuracy
    """
    tokenizer, model_rec, model_share, device = load_models_and_tokenizer()

    # Load checkpoint
    if verbose:
        print(f"Loading checkpoint from step {checkpoint_step}...")
    c2c_model, _, metadata = load_checkpoint(
        step=checkpoint_step,
        ckpt_dir=ckpt_dir,
        model_rec=model_rec,
        model_share=model_share,
        tokenizer=tokenizer,
    )
    c2c_model.eval()
    c2c_model.c2c.sanity_check = True  # Enable sanity check to just evaluate the receiver raw model

    if verbose:
        print(f"Checkpoint loaded: step={metadata['step']}, loss={metadata['loss']:.4f}")
        print(f"Evaluating on {len(categories or MMLU_REDUX_CATEGORIES)} categories...\n")

    # Use all categories if not specified
    if categories is None:
        categories = MMLU_REDUX_CATEGORIES

    # Results tracking
    all_count_correct = {}
    all_count_incorrect = {}
    g_count_correct = 0
    g_count_incorrect = 0

    # Evaluate each category
    for category in tqdm(categories, desc="Category", disable=verbose):
        try:
            ds = load_dataset("edinburgh-dawg/mmlu-redux", category)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load category '{category}': {e}")
            continue

        count_correct = 0
        count_incorrect = 0

        num_questions = len(ds["test"])
        if max_questions_per_category is not None:
            num_questions = min(num_questions, max_questions_per_category)

        for idx in tqdm(range(num_questions), desc=f"  {category}", leave=False, disable=verbose):
            try:
                question, correct_answer = get_mmlu_question(ds, idx)
                if question is None:
                    continue
                prompt = get_chat_template(question, tokenizer)

                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                with torch.no_grad():
                    g = c2c_model.generate(
                        prompt_input_ids=inputs["input_ids"],
                        prompt_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        tokenizer=tokenizer,
                    )
                generated_ids = g[0].tolist()
                answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                if check_answer(correct_answer, answer_text):
                    count_correct += 1
                    g_count_correct += 1
                    str_judgement = "Correct"
                else:
                    count_incorrect += 1
                    g_count_incorrect += 1
                    str_judgement = "Incorrect"

                if verbose:
                    category_acc = count_correct / (count_correct + count_incorrect)
                    global_acc = g_count_correct / (g_count_correct + g_count_incorrect)
                    print(
                        f"\n[{idx}/{num_questions}] Ans: {answer_text}\t\t Judge: {str_judgement}\t\t"
                        f"<Accuracy> {category}: {category_acc:.2%}, "
                        f"global: {global_acc:.2%}"
                    )

            except KeyboardInterrupt:
                if verbose:
                    print("\nBenchmark interrupted by user.")
                break

        all_count_correct[category] = count_correct
        all_count_incorrect[category] = count_incorrect

        if verbose:
            category_acc = count_correct / (count_correct + count_incorrect) if (count_correct + count_incorrect) > 0 else 0.0
            print(f"\n{category} final accuracy: {category_acc:.2%} ({count_correct}/{count_correct + count_incorrect})\n")

    # Calculate final results
    total_questions = g_count_correct + g_count_incorrect
    global_accuracy = g_count_correct / total_questions if total_questions > 0 else 0.0

    results = {
        "checkpoint_step": checkpoint_step,
        "checkpoint_metadata": metadata,
        "categories": categories,
        "per_category_correct": all_count_correct,
        "per_category_incorrect": all_count_incorrect,
        "global_correct": g_count_correct,
        "global_incorrect": g_count_incorrect,
        "global_accuracy": global_accuracy,
        "total_questions": total_questions,
    }

    if verbose:
        print("=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Checkpoint: step {checkpoint_step}")
        print(f"Global Accuracy: {global_accuracy:.2%} ({g_count_correct}/{total_questions})")
        print("\nPer-category accuracy:")
        for category in categories:
            if category in all_count_correct:
                correct = all_count_correct[category]
                incorrect = all_count_incorrect[category]
                acc = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0.0
                print(f"  {category}: {acc:.2%} ({correct}/{correct + incorrect})")
        print("=" * 60)

    return results


if __name__ == "__main__":
    benchmark(
        checkpoint_step=8,
        categories=None,
        max_questions_per_category=10,
        max_new_tokens=7,
    )
