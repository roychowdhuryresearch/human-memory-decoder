import torch
import torch.nn as nn


class Ensemble(nn.Module):
    def __init__(self, lfp_model, spike_model, config):
        super().__init__()
        self.spike_model = spike_model
        self.lfp_model = lfp_model

        if self.lfp_model and not self.spike_model:
            self.mlp_head = nn.Linear(config['hidden_size'], config['num_labels'])
        elif not self.lfp_model and self.spike_model:
            self.mlp_head = nn.Linear(config['hidden_size'], config['num_labels'])
            # self.mlp_head = nn.Identity()

        self.device = config['device']

    def forward(self, lfp, spike):
        if self.spike_model and not self.lfp_model:
            spike_emb = self.spike_model(spike, output_attentions=True)
            lfp_emb = None
            combined_emb = spike_emb[0]
            attentions = spike_emb[1]
            combined_emb = self.mlp_head(combined_emb)
        elif not self.spike_model and self.lfp_model:
            spike_emb = None
            lfp_emb = self.lfp_model(lfp)
            combined_emb = lfp_emb[0]
            attentions = lfp_emb[1]
            combined_emb = self.mlp_head(combined_emb)

        # combined_emb = self.sigmoid(combined_emb)
        return spike_emb, lfp_emb, combined_emb, attentions





