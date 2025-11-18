import torch

class RPX:
    def __init__(self, model, refresh_steps=4):
        self.mirror = model.base_model  # frozen reference
        self.refresh_steps = refresh_steps
        self.step = 0

    @torch.no_grad()
    def delta_c(self, hidden_e, hidden_m, pred_e, pred_m):
        cos = torch.nn.functional.cosine_similarity(hidden_e, hidden_m, dim=-1).mean()
        pred_diff = torch.abs(pred_e - pred_m).mean()
        return pred_diff * (1 - cos)

    def forward_with_mirror(self, input_ids):
        # Evolving forward
        outputs_e = model(input_ids, output_hidden_states=True)
        hidden_e = outputs_e.hidden_states[-1][:, -1, :]

        # Mirror forward (same weights если step==0, иначе frozen)
        if self.step % self.refresh_steps == 0:
            self.mirror.load_state_dict(model.base_model.state_dict())
        outputs_m = self.mirror(input_ids, output_hidden_states=True)
        hidden_m = outputs_m.hidden_states[-1][:, -1, :]

        delta = self.delta_c(hidden_e, hidden_m, outputs_e.logits, outputs_m.logits)
        self.step += 1
        return outputs_e, delta.item()
