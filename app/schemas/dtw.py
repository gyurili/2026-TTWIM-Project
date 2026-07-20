from typing import Annotated

from pydantic import Field

from app.schemas.base import CamelModel


class DtwData(CamelModel):
    axis_name: str
    template: Annotated[list[float], Field(min_length=1)]
    base_dist: float
    dist_mad: float
    baseline_cadence: float
    cadence_mad: float
    baseline_amplitude: float
    amplitude_mad: float
