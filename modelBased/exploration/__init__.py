"""Exploration policies used to acquire world-model training data."""

__all__ = [
    "CrafterP2EExplorer",
    "P2EEnsemble",
    "DreamerP2EActorCritic",
    "AttentionWMP2EAdapter",
    "EpisodeReplay",
    "CrafterStateActionCounter",
    "MiniGridSemanticNoveltyCounter",
    "MiniGridRMaxExplorer",
    "MiniGridDQNExplorer",
]


def __getattr__(name):
    """Keep count acquisition from importing the P2E/WM-facing stack."""
    if name in {"CrafterStateActionCounter", "MiniGridSemanticNoveltyCounter"}:
        from .count_based import CrafterStateActionCounter, MiniGridSemanticNoveltyCounter

        return {
            "CrafterStateActionCounter": CrafterStateActionCounter,
            "MiniGridSemanticNoveltyCounter": MiniGridSemanticNoveltyCounter,
        }[name]
    if name == "MiniGridRMaxExplorer":
        from .minigrid_rmax import MiniGridRMaxExplorer

        return MiniGridRMaxExplorer
    if name == "MiniGridDQNExplorer":
        from .minigrid_dqn import MiniGridDQNExplorer

        return MiniGridDQNExplorer
    if name in {"CrafterP2EExplorer", "P2EEnsemble", "DreamerP2EActorCritic", "AttentionWMP2EAdapter", "EpisodeReplay"}:
        from .p2e import AttentionWMP2EAdapter, CrafterP2EExplorer, DreamerP2EActorCritic, EpisodeReplay, P2EEnsemble

        return {
            "CrafterP2EExplorer": CrafterP2EExplorer,
            "P2EEnsemble": P2EEnsemble,
            "DreamerP2EActorCritic": DreamerP2EActorCritic,
            "AttentionWMP2EAdapter": AttentionWMP2EAdapter,
            "EpisodeReplay": EpisodeReplay,
        }[name]
    raise AttributeError(name)
