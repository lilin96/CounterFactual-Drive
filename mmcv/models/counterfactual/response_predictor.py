"""Neural relevance and response meta-action predictors."""

from torch import nn


class ResponsePredictor(nn.Module):
    """Predict interaction relevance plus speed/path response logits."""

    def __init__(self, hidden_dims=256, num_speed=7, num_path=6):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(inplace=True),
        )
        self.rel_head = nn.Linear(hidden_dims, 1)
        self.speed_head = nn.Linear(hidden_dims, num_speed)
        self.path_head = nn.Linear(hidden_dims, num_path)

    def forward(self, interaction_features):
        h = self.shared(interaction_features)
        return dict(
            relevance_logits=self.rel_head(h).squeeze(-1),
            speed_logits=self.speed_head(h),
            path_logits=self.path_head(h),
        )

