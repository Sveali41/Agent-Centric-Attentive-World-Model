"""Exploration policies used to acquire world-model training data."""

__all__ = ["CrafterP2EExplorer", "P2EEnsemble", "DreamerP2EActorCritic", "AttentionWMP2EAdapter", "EpisodeReplay", "CrafterStateActionCounter"]


def __getattr__(name):
    """Keep count acquisition from importing the P2E/WM-facing stack."""
    if name == "CrafterStateActionCounter":
        from .count_based import CrafterStateActionCounter

        return CrafterStateActionCounter
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
