# =============================================================
# Cache-to-Cache (C2C) Training Script
# =============================================================

import torch
import mlflow
import mlflow.pytorch
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

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("c2c_training")
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(
            {
                "total_steps": total_steps,
                "learning_rate": lr,
                "temp_start": temp_start,
                "temp_end": temp_end,
                "save_every": save_every,
            }
        )

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

            # Forward pass with debug info
            loss, debug_info = c2c_model(p_ids, p_mask, r_ids, r_mask, labels, return_debug=True)
            loss.backward()

            # Get gate parameters and gradients
            gate_param = c2c_model.c2c.gate_param.detach().cpu()
            gate_param_grad = c2c_model.c2c.gate_param.grad
            gate_param_grad_norm = gate_param_grad.norm().item() if gate_param_grad is not None else 0.0

            # Compute inference gate behavior (what would be used at inference time)
            inference_gates = (gate_param > 0).float()
            num_active_gates = inference_gates.sum().item()
            active_gate_ratio = num_active_gates / len(gate_param)

            # Get parameter norms
            param_norms = {}
            for name, param in c2c_model.c2c.named_parameters():
                if param.grad is not None:
                    param_norms[f"grad_norm_{name}"] = param.grad.norm().item()
                param_norms[f"param_norm_{name}"] = param.norm().item()

            # Get actual training gate values
            gate_values = debug_info["gate_values"]
            avg_gate_value = sum(gate_values) / len(gate_values) if gate_values else 0.0

            # Compute fused cache contribution
            fused_K_norms = debug_info.get("fused_K_norms", [])
            fused_V_norms = debug_info.get("fused_V_norms", [])
            K_R_norms = debug_info.get("K_R_norms", [])
            V_R_norms = debug_info.get("V_R_norms", [])

            avg_fused_K_norm = sum(fused_K_norms) / len(fused_K_norms) if fused_K_norms else 0.0
            avg_fused_V_norm = sum(fused_V_norms) / len(fused_V_norms) if fused_V_norms else 0.0
            avg_K_R_norm = sum(K_R_norms) / len(K_R_norms) if K_R_norms else 0.0
            avg_V_R_norm = sum(V_R_norms) / len(V_R_norms) if V_R_norms else 0.0

            optimizer.step()

            # Prepare metrics for logging
            metrics = {
                "loss": loss.item(),
                "temperature": temperature,
                # Gate parameter statistics
                "gate_param_mean": gate_param.mean().item(),
                "gate_param_min": gate_param.min().item(),
                "gate_param_max": gate_param.max().item(),
                "gate_param_std": gate_param.std().item(),
                # Inference gate behavior (critical for debugging!)
                "inference_active_gates": num_active_gates,
                "inference_active_gate_ratio": active_gate_ratio,
                # Training gate values
                "avg_training_gate_value": avg_gate_value,
                "min_training_gate_value": min(gate_values) if gate_values else 0.0,
                "max_training_gate_value": max(gate_values) if gate_values else 0.0,
                # Gradient information
                "gate_param_grad_norm": gate_param_grad_norm,
                # Fused cache contribution
                "avg_fused_K_norm": avg_fused_K_norm,
                "avg_fused_V_norm": avg_fused_V_norm,
                "avg_K_R_norm": avg_K_R_norm,
                "avg_V_R_norm": avg_V_R_norm,
                "fused_K_to_K_R_ratio": avg_fused_K_norm / avg_K_R_norm if avg_K_R_norm > 0 else 0.0,
                "fused_V_to_V_R_ratio": avg_fused_V_norm / avg_V_R_norm if avg_V_R_norm > 0 else 0.0,
            }

            # Add parameter and gradient norms
            metrics.update(param_norms)

            # Log individual gate parameters
            for i in range(len(gate_param)):
                metrics[f"gate_param_layer_{i}"] = gate_param[i].item()
                if i < len(gate_values):
                    metrics[f"training_gate_layer_{i}"] = gate_values[i]
                metrics[f"inference_gate_layer_{i}"] = inference_gates[i].item()

            # Log all metrics to MLflow
            mlflow.log_metrics(metrics, step=step)

            print(
                f"step {step} | loss {loss.item():.4f} | temp {temperature:.4f} | "
                f"gate_mean {gate_param.mean().item():.4f} | "
                f"active_gates {num_active_gates}/{len(gate_param)} ({active_gate_ratio:.2%}) | "
                f"avg_training_gate {avg_gate_value:.4f} | "
                f"grad_norm {gate_param_grad_norm:.6f}"
            )
            if step % save_every == 0:
                save_checkpoint(step, c2c_model, optimizer, loss.item())

        # Save final checkpoint (use total_steps as step number for clarity)
        if (total_steps - 1) % save_every != 0:
            save_checkpoint(total_steps, c2c_model, optimizer, loss.item())


if __name__ == "__main__":
    train(total_steps=10000)
