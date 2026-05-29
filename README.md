# Standalone SCALE Usage

`scale_optimizer_standalone.py` decouples the SCALE update from any specific model, dataset, or task pipeline. It only depends on standard PyTorch objects: `model`, `optimizer`, `loss_closure`, and an externally provided scalar `scale_weight`. For model-related code, including the model definition, data format, loss function, and original training loop, please directly refer to the **SafeDrug GitHub repository**.

## 1. Core Update

The SCALE update is implemented as:

```text
g = gamma * grad L(theta + epsilon)
    + grad[(1 - gamma * scale_weight) * L(theta)]
```

where:

- `L(theta)` is the ordinary training loss for the current sample or batch;
- `epsilon = rho * grad L(theta) / ||grad L(theta)||` is a SAM-style parameter perturbation;
- `scale_weight` is an external scalar weight, such as a sample weight, group weight, or tail/head weight;
- `gamma` controls the strength of the sharpness gradient and the scale-aware correction;
- `rho` controls the perturbation radius.

## 2. Recommended Usage

```python
from scale_optimizer_standalone import ScaleConfig, scale_step

scale_cfg = ScaleConfig(
    rho=0.05,
    gamma=1.0,
    max_grad_norm=1.0,
)

for batch in dataloader:
    def loss_closure():
        # Write your own forward and loss computation here.
        # The closure must return a scalar loss and must not call backward().
        output = model(batch["x"])
        loss = criterion(output, batch["y"])
        return loss

    info = scale_step(
        model=model,
        optimizer=optimizer,
        loss_closure=loss_closure,
        scale_weight=batch["weight"],
        config=scale_cfg,
    )
```

`scale_step()` automatically performs the following operations:

1. computes the clean loss and clean gradient;
2. builds a SAM-style perturbation from the clean gradient;
3. recomputes the loss at the perturbed parameters;
4. computes the sharpness gradient;
5. computes the scale-aware clean gradient;
6. combines the final SCALE gradient;
7. optionally applies global gradient clipping;
8. writes the manual gradient into `param.grad` and calls `optimizer.step()`.

## 3. Integration with the Original SafeDrug Training Code

The model-specific loss remains in the main training script. SCALE only controls the optimization step.

```python
from scale_optimizer_standalone import ScaleConfig, scale_step

scale_cfg = ScaleConfig(
    rho=config["rho"],
    gamma=config["gamma"],
    max_grad_norm=1.0,
)

for step, input in enumerate(data_train):
    for idx, adm in enumerate(input):
        seq_input = input[: idx + 1]

        def loss_closure():
            loss, _ = compute_loss(
                model=model,
                seq_input=seq_input,
                adm=adm,
                voc_size=voc_size,
                ddi_flag=config["ddi_flag"],
                device=device,
                ddi_adj_path=ddi_adj_path,
                config=config,
            )
            return loss

        scale_step(
            model=model,
            optimizer=optimizer,
            loss_closure=loss_closure,
            scale_weight=adm[5],
            config=scale_cfg,
        )
```

In this example, `compute_loss`, `seq_input`, `adm`, `voc_size`, and DDI-related arguments are still task-specific. The SCALE module does not inspect or depend on them.

## 4. Minimal Sanity Test

```python
import torch
from torch import nn
from torch.optim import Adam
from scale_optimizer_standalone import ScaleConfig, scale_step

model = nn.Linear(4, 2)
optimizer = Adam(model.parameters(), lr=1e-3)
x = torch.randn(8, 4)
y = torch.randn(8, 2)
criterion = nn.MSELoss()

cfg = ScaleConfig(rho=0.05, gamma=1.0, max_grad_norm=1.0)

def closure():
    return criterion(model(x), y)

info = scale_step(
    model=model,
    optimizer=optimizer,
    loss_closure=closure,
    scale_weight=0.8,
    config=cfg,
)

print(info)
```
