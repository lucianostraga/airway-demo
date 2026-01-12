"""
ULD (Unit Load Device) domain models.

ULDs are standardized containers used to load luggage, cargo, and mail on aircraft.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class ULDType(str, Enum):
    """
    IATA ULD type codes.

    Each type has specific dimensions, weight limits, and aircraft compatibility.
    """

    # Lower deck containers (narrow-body compatible)
    AKE = "AKE"  # LD3 - Most common, fits A320/A330/A350/B777/B787
    AKH = "AKH"  # LD3-45 - Similar to AKE with different door
    DPE = "DPE"  # LD2 - Smaller, for regional jets

    # Lower deck containers (wide-body only)
    AKN = "AKN"  # LD7 - Larger, B747/B777/A380
    AAP = "AAP"  # LD9 - B747/B777
    ALP = "ALP"  # LD6 - B747/B777

    # Pallets
    PMC = "PMC"  # P6P - Standard pallet with net
    PAG = "PAG"  # P1P - 20ft pallet for freighters
    PLA = "PLA"  # P1P variant

    # Special containers
    RKN = "RKN"  # Temperature controlled (reefer)
    HMA = "HMA"  # Horse container
    AKC = "AKC"  # Animal container


class ULDStatus(str, Enum):
    """ULD operational status."""

    SERVICEABLE = "serviceable"  # Available for use
    IN_USE = "in_use"  # Currently loaded/in transit
    EMPTY = "empty"  # At station, available for loading
    DAMAGED = "damaged"  # Requires inspection
    OUT_OF_SERVICE = "out_of_service"  # At MRO for repair
    IN_TRANSIT = "in_transit"  # On aircraft between stations
    CUSTOMS_HOLD = "customs_hold"  # Held for customs clearance


class ULDSpecification(BaseModel):
    """Physical specifications for a ULD type."""

    uld_type: ULDType
    max_gross_weight_kg: int
    tare_weight_kg: int
    internal_volume_m3: float
    max_height_cm: int
    max_width_cm: int
    max_length_cm: int

    @property
    def max_payload_kg(self) -> int:
        return self.max_gross_weight_kg - self.tare_weight_kg


# Standard ULD specifications (IATA standards)
ULD_SPECS: dict[ULDType, ULDSpecification] = {
    ULDType.AKE: ULDSpecification(
        uld_type=ULDType.AKE,
        max_gross_weight_kg=1588,
        tare_weight_kg=83,
        internal_volume_m3=4.3,
        max_height_cm=163,
        max_width_cm=153,
        max_length_cm=156,
    ),
    ULDType.AKH: ULDSpecification(
        uld_type=ULDType.AKH,
        max_gross_weight_kg=1588,
        tare_weight_kg=85,
        internal_volume_m3=4.3,
        max_height_cm=163,
        max_width_cm=153,
        max_length_cm=156,
    ),
    ULDType.PMC: ULDSpecification(
        uld_type=ULDType.PMC,
        max_gross_weight_kg=6804,
        tare_weight_kg=120,
        internal_volume_m3=11.5,
        max_height_cm=163,
        max_width_cm=244,
        max_length_cm=318,
    ),
    ULDType.AKN: ULDSpecification(
        uld_type=ULDType.AKN,
        max_gross_weight_kg=5035,
        tare_weight_kg=170,
        internal_volume_m3=9.9,
        max_height_cm=163,
        max_width_cm=244,
        max_length_cm=317,
    ),
}


class ULD(BaseModel):
    """
    Unit Load Device entity.

    Represents a single ULD container with its current state.
    """

    uld_id: Annotated[str, Field(description="Unique ULD identifier (e.g., AKE12345DL)")]
    uld_type: ULDType
    owner_airline: str = Field(default="DL", pattern=r"^[A-Z]{2}$")
    status: ULDStatus = ULDStatus.SERVICEABLE
    current_station: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    last_known_position: "ULDPosition | None" = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"frozen": False}

    @field_validator("uld_id")
    @classmethod
    def validate_uld_id(cls, v: str) -> str:
        """Validate ULD ID format: TYPE + SERIAL + OWNER (e.g., AKE12345DL)."""
        if len(v) < 8:
            raise ValueError("ULD ID must be at least 8 characters")
        return v.upper()

    @property
    def spec(self) -> ULDSpecification | None:
        """Get specifications for this ULD type."""
        return ULD_SPECS.get(self.uld_type)

    def is_available(self) -> bool:
        """Check if ULD is available for loading."""
        return self.status in (ULDStatus.SERVICEABLE, ULDStatus.EMPTY)


class ULDPosition(BaseModel):
    """
    ULD position tracking record.

    Represents a geolocation update from tracking device.
    """

    uld_id: str
    uld_type: "ULDType"
    station: str = Field(pattern=r"^[A-Z]{3}$")
    timestamp: datetime
    position_source: str = "geolocation"  # geolocation, flight_event, flight_inference
    flight_number: str | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    location_type: str = "ground"  # ground, aircraft, warehouse
    confidence: float = Field(default=1.0, ge=0, le=1)

    model_config = {"frozen": True}


class ULDInventory(BaseModel):
    """
    ULD inventory at a station by type.

    Snapshot of ULD availability at a specific location and time.
    """

    station: str = Field(..., pattern=r"^[A-Z]{3}$")
    timestamp: datetime
    inventory: dict[ULDType, int] = Field(default_factory=dict)
    available: dict[ULDType, int] = Field(default_factory=dict)
    in_use: dict[ULDType, int] = Field(default_factory=dict)
    damaged: dict[ULDType, int] = Field(default_factory=dict)

    model_config = {"frozen": True}

    def total_count(self) -> int:
        """Total ULDs at station."""
        return sum(self.inventory.values())

    def total_available(self) -> int:
        """Total available ULDs."""
        return sum(self.available.values())

    def availability_ratio(self) -> float:
        """Ratio of available to total ULDs."""
        total = self.total_count()
        return self.total_available() / total if total > 0 else 0.0

    def get_available(self, uld_type: ULDType) -> int:
        """Get available count for a specific type."""
        return self.available.get(uld_type, 0)
