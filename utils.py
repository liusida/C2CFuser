# =============================================================
# C2C Utility Functions
# =============================================================

import os
import json
import torch
from safetensors.torch import save_file, load_file
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from models import C2CReceiverForCausalLM


# -------------------------------------------------------------
# Device & Model Configuration
# -------------------------------------------------------------


def get_device():
    """Get the device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_models_and_tokenizer(
    model_rec_name="Qwen/Qwen3-0.6B",
    model_share_name="Qwen/Qwen3-1.7B",
    device=None,
):
    """
    Load models and tokenizer.

    Args:
        model_rec_name: Name of the receiver model
        model_share_name: Name of the sharer model
        device: Device to load models on (None = auto-detect)

    Returns:
        tuple: (tokenizer, model_rec, model_share, device)
    """
    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_rec_name)

    model_rec = AutoModelForCausalLM.from_pretrained(
        model_rec_name,
        torch_dtype=torch.float16,
        device_map=None,
    ).to(device)

    model_share = AutoModelForCausalLM.from_pretrained(
        model_share_name,
        torch_dtype=torch.float16,
        device_map=None,
    ).to(device)

    return tokenizer, model_rec, model_share, device


# -------------------------------------------------------------
# Dataset Utilities
# -------------------------------------------------------------


def load_training_dataset(dataset_name="teknium/OpenHermes-2.5"):
    """Load the training dataset."""
    return load_dataset(dataset_name)


# -------------------------------------------------------------
# Conversation Utilities
# -------------------------------------------------------------


def split_prompt_response_by_think(text, marker="</think>"):
    """Split text by marker, returning prompt and response parts."""
    idx = text.rfind(marker)
    if idx == -1:
        print(f"Warning: Marker {marker} not found")
        return None, None
    return text[: idx + len(marker)], text[idx + len(marker) :]


def get_formatted_conversation(conversation, tokenizer):
    """Format conversation using tokenizer's chat template."""
    messages = []
    for c in conversation:
        role = c["from"]
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        elif role != "system":
            print(f"Warning: unknown role {role}")
            break
        messages.append({"role": role, "content": c["value"]})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


# -------------------------------------------------------------
# Checkpoint Utilities
# -------------------------------------------------------------


def save_checkpoint(step, c2c_model, optimizer, loss, ckpt_dir="checkpoints"):
    """Save checkpoint with model weights, optimizer state, and metadata."""
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_base = os.path.join(ckpt_dir, f"ckpt_step_{step}")
    safetensors_path = f"{ckpt_base}.safetensors"
    metadata_path = f"{ckpt_base}_metadata.json"

    # Save state dict as safetensors
    save_file(c2c_model.c2c.state_dict(), safetensors_path)

    # Save optimizer state
    optimizer_path = f"{ckpt_base}_optimizer.pt"
    torch.save(optimizer.state_dict(), optimizer_path)

    # Save metadata as JSON (including learning rate from optimizer)
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "step": step,
                "loss": float(loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
            },
            f,
            indent=2,
        )

    return safetensors_path, metadata_path


def load_checkpoint(step, ckpt_dir="checkpoints", model_rec=None, model_share=None, tokenizer=None):
    """Load checkpoint with model weights, optimizer state, and metadata."""
    ckpt_base = os.path.join(ckpt_dir, f"ckpt_step_{step}")
    safetensors_path = f"{ckpt_base}.safetensors"
    metadata_path = f"{ckpt_base}_metadata.json"
    optimizer_path = f"{ckpt_base}_optimizer.pt"

    # Create model
    if model_rec is None or model_share is None:
        raise ValueError("model_rec and model_share must be provided")

    c2c_model = C2CReceiverForCausalLM(model_rec, model_share, tokenizer=tokenizer)

    # Load model weights from safetensors
    state_dict = load_file(safetensors_path)
    c2c_model.c2c.load_state_dict(state_dict)

    # Load metadata first to get learning rate
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Create optimizer with saved learning rate and load its state
    lr = metadata.get("lr", 1e-4)  # fallback to default if not in metadata
    optimizer = torch.optim.AdamW(c2c_model.c2c.parameters(), lr=lr)
    optimizer.load_state_dict(torch.load(optimizer_path))

    return c2c_model, optimizer, metadata


# -------------------------------------------------------------
# Benchmark Utilities
# -------------------------------------------------------------

MMLU_REDUX_CATEGORIES = [
    "anatomy",
    "business_ethics",
    "clinical_knowledge",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "econometrics",
    "electrical_engineering",
    "formal_logic",
    "global_facts",
    "high_school_chemistry",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "human_aging",
    "logical_fallacies",
    "machine_learning",
    "miscellaneous",
    "philosophy",
    "professional_accounting",
    "public_relations",
    "virology",
    "conceptual_physics",
    "high_school_us_history",
    "astronomy",
    "high_school_geography",
    "high_school_macroeconomics",
    "professional_law",
]


def get_chat_template(question, tokenizer):
    """Format question using tokenizer's chat template."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def get_mmlu_question(ds, idx):
    """Extract question and correct answer from MMLU Redux dataset."""
    question = f"""Accurately answer the following question, respond with a single letter of the correct answer:
{ds['test'][idx]['question']}

Choices:
A. {ds['test'][idx]['choices'][0]}
B. {ds['test'][idx]['choices'][1]}
C. {ds['test'][idx]['choices'][2]}
D. {ds['test'][idx]['choices'][3]}
"""

    correct_answer_idx = ds["test"][idx]["answer"]
    if ds["test"][idx].get("correct_answer") is not None:
        correct_answer_idx = ds["test"][idx]["correct_answer"]

    try:
        correct_answer_idx = int(correct_answer_idx)
        correct_answer = "ABCD"[correct_answer_idx]
    except:
        print(f"Bad: {ds['test'][idx]}")
        return None, None

    return question, correct_answer


def check_answer(correct_answer, answer_text):
    """Check if the generated answer matches the correct answer."""
    if f"{correct_answer}." in answer_text:
        if answer_text.strip().startswith(correct_answer):
            return True
        else:
            print(f"Warning: this answer might be wrong: {answer_text}\n\n")
            return True
    return False
