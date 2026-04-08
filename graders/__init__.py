from graders.base_grader import BaseGrader, GraderTier
from graders.easy_grader import EasyGrader
from graders.medium_grader import MediumGrader
from graders.hard_grader import HardGrader


class GraderRegistry:
    _registry = {
        GraderTier.EASY: EasyGrader,
        GraderTier.MEDIUM: MediumGrader,
        GraderTier.HARD: HardGrader,
    }

    @classmethod
    def get(cls, tier: GraderTier, **kwargs) -> BaseGrader:
        klass = cls._registry.get(tier)
        if klass is None:
            raise ValueError(f"No grader registered for tier: {tier}")
        return klass(**kwargs)

    @classmethod
    def from_string(cls, tier_str: str, **kwargs) -> BaseGrader:
        mapping = {
            "easy": GraderTier.EASY,
            "medium": GraderTier.MEDIUM,
            "hard": GraderTier.HARD,
        }
        tier = mapping.get(tier_str.lower())
        if tier is None:
            raise ValueError(
                f"Unknown grader tier '{tier_str}'. Valid values: {list(mapping.keys())}"
            )
        return cls.get(tier, **kwargs)


__all__ = ["BaseGrader", "GraderTier", "EasyGrader", "MediumGrader", "HardGrader", "GraderRegistry"]