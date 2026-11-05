"""What every skill returns."""
from dataclasses import dataclass, field


@dataclass
class SkillResult:
    success: bool
    message: str = ""
    info: dict = field(default_factory=dict)

    def __bool__(self):
        return self.success
