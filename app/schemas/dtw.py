from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class DtwData(BaseModel):
    axis_name: Annotated[str, Field(alias="axisName")]
    template: Annotated[list[float], Field(min_length=1)]
    base_dist: Annotated[float, Field(alias="baseDist")]
    dist_mad: Annotated[float, Field(alias="distMad")]
    baseline_cadence: Annotated[float, Field(alias="baselineCadence")]
    cadence_mad: Annotated[float, Field(alias="cadenceMad")]
    baseline_amplitude: Annotated[float, Field(alias="baselineAmplitude")]
    amplitude_mad: Annotated[float, Field(alias="amplitudeMad")]

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)
