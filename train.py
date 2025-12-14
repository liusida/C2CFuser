# =============================================================
# Cache-to-Cache (C2C) Training Script
# =============================================================

import torch
from models import C2CReceiverForCausalLM
from utils import (
    load_models_and_tokenizer,
    load_training_dataset,
    split_prompt_response_by_think,
    get_formatted_conversation,
    save_checkpoint,
)


# -------------------------------------------------------------
# Training Loop
# -------------------------------------------------------------
def train(total_steps=10000):
    """
    Train the C2C model.

    Args:
        total_steps: Total number of training steps to run
    """
    tokenizer, model_rec, model_share, device = load_models_and_tokenizer()
    dataset = load_training_dataset()
    lr = 1e-4
    c2c_model = C2CReceiverForCausalLM(model_rec, model_share, tokenizer=tokenizer)
    c2c_model.train()
    optimizer = torch.optim.AdamW(c2c_model.c2c.parameters(), lr=lr)
    save_every = 1000
    temp_start = 1.0
    temp_end = 0.05

    for step in range(total_steps):
        # ---- compute and set temperature (ONE LINE OF CONTROL) ----
        progress = step / max(total_steps - 1, 1)
        temperature = temp_start * (1 - progress) + temp_end * progress
        c2c_model.c2c.set_temperature(temperature)

        conv = dataset["train"][step]["conversations"]
        prompt = get_formatted_conversation(conv, tokenizer)
        prompt_text, response_text = split_prompt_response_by_think(prompt)

        if prompt_text is None:
            continue

        optimizer.zero_grad(set_to_none=True)

        p_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        r_enc = tokenizer(response_text, return_tensors="pt", add_special_tokens=False)

        p_ids = p_enc.input_ids.to(device)
        p_mask = p_enc.attention_mask.to(device)
        r_ids = r_enc.input_ids.to(device)
        r_mask = r_enc.attention_mask.to(device)

        labels = r_ids.clone()
        labels[r_mask == 0] = -100

        loss = c2c_model(p_ids, p_mask, r_ids, r_mask, labels)
        loss.backward()
        optimizer.step()

        print(f"step {step} | loss {loss.item():.4f}")
        if step % save_every == 0:
            save_checkpoint(step, c2c_model, optimizer, loss.item())

    # Save final checkpoint (use total_steps as step number for clarity)
    if (total_steps - 1) % save_every != 0:
        save_checkpoint(total_steps, c2c_model, optimizer, loss.item())


if __name__ == "__main__":
    train(total_steps=8)
