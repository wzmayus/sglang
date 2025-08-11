# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from dataclasses import dataclass, field, fields
from typing import Optional
from uuid import uuid4


@dataclass(frozen=True)
class LoRARef:
    """
    Reference record for a LoRA model.

    This object guarantees a unique ``lora_id`` and may include ``lora_name``, ``lora_path``, and ``pinned``.
    The ID eliminates conflicts from reused LoRA names or paths and can be used to generate deterministic cache
    keys (e.g., radix cache).
    """

    lora_id: str = field(default_factory=lambda: uuid4().hex)
    lora_name: Optional[str] = None
    lora_path: Optional[str] = None
    pinned: Optional[bool] = None

    def __post_init__(self):
        if self.lora_id is None:
            raise ValueError("lora_id cannot be None")

    def __str__(self) -> str:
        parts = [
            f"{f.name}={value}"
            for f in fields(self)
            if (value := getattr(self, f.name)) is not None
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"
