"""Knobs of the default perception pipeline (`configs/perception.example.yaml`).

Defaults assume a tabletop scene seen by a calibrated RGB-D camera with metric depth. Where the
table is (`table.xy_box`, `table.z_range`) and which camera anchors a frame (`anchor`) have no
defaults: a benchmark's or robot's perception binding sets them. Lengths are meters; pixel counts
and 8-bit color levels are integers.
"""

from typing import Final

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat, model_validator
from typing_extensions import Self

_Unit = ConfigDict(frozen=True, extra="forbid")


def _ordered(values: tuple[float, ...], name: str) -> None:
    for lo, hi in zip(values[::2], values[1::2], strict=True):
        if not lo < hi:
            raise ValueError(f"{name} needs lower < upper bounds, got {values}")


class TableConfig(BaseModel):
    """Where the table is searched for and how its plane is fit (`perception.table`)."""

    model_config = _Unit

    xy_box: tuple[float, float, float, float] | None = None
    """x0, x1, y0, y1: the nominal table area; only points inside vote for the table height.
    None: unset, and finding a table refuses (`area`)."""
    z_range: tuple[float, float] | None = None
    """z0, z1: the heights searched for the table plane; None: unset (`heights`)."""
    band: PositiveFloat = 0.003
    """Half width of the band around the 1 mm height-histogram peak that is refit as a plane."""
    inlier: PositiveFloat = 0.001
    """Plane distance of the points that define the table's extent."""
    max_tilt_deg: PositiveFloat = 2.0
    min_points: int = Field(default=500, ge=3)
    """A band fit tilted more than `max_tilt_deg`, or with fewer points, falls back to RANSAC."""
    ransac_threshold: PositiveFloat = 0.002
    drop: PositiveFloat = 0.01
    """A table pixel next to a pixel this far below the plane (the floor, the wall) is on one of
    the table's edges."""
    edge_points: int = Field(default=30, ge=1)
    """An edge needs this many edge pixels (all views together)."""

    @model_validator(mode="after")
    def _ranges(self) -> Self:
        if self.xy_box is not None:
            _ordered(self.xy_box, "xy_box")
        if self.z_range is not None:
            _ordered(self.z_range, "z_range")
        return self

    @property
    def area(self) -> tuple[float, float, float, float]:
        """`xy_box`, which must be set."""
        if self.xy_box is None:
            raise ValueError(
                "perception.table.xy_box is unset: set the nominal table area for the scene "
                "(a benchmark's perception binding does)"
            )
        return self.xy_box

    @property
    def heights(self) -> tuple[float, float]:
        """`z_range`, which must be set."""
        if self.z_range is None:
            raise ValueError(
                "perception.table.z_range is unset: set the heights the table is searched at "
                "(a benchmark's perception binding does)"
            )
        return self.z_range


class LiftConfig(BaseModel):
    """Turning masks into an object's world points (`perception.lift`)."""

    model_config = _Unit

    support_height: PositiveFloat = 0.0005
    """Points this close to the table plane are the table itself, at least: on noisier depth,
    those within the table's own noise (`perception.lift.weak_height`)."""
    flat_band: PositiveFloat = 0.01
    """A mask with (almost) nothing standing above the proposals' `min_height` is a flat patch
    (a 1-2 mm mat): it keeps its points within this band of the table."""
    floor_z: float = 0.005
    """Points below this height are the floor."""
    wall_margin: NonNegativeFloat | None = 0.03
    """Points this far beyond the table's far edge (or the anchor view's extent when the edge is
    unknown) are background (a wall, say); None keeps them."""
    min_points: int = Field(default=10, ge=1)
    view_gap: PositiveFloat = 0.02
    view_share: float = Field(default=0.2, gt=0, le=1)
    """A non-anchor view's points join the anchor view's when this share of them comes within
    `view_gap` of them."""
    voxel: PositiveFloat = 0.002
    """Points merged from several views are averaged per cube of this edge."""


class ProposalConfig(BaseModel):
    """Object proposals on the table (`perception.proposals`)."""

    model_config = _Unit

    min_height: PositiveFloat = 0.004
    """Height-map cells need points this far above the table (a real sensor resolves ~4 mm)."""
    min_cells: int = Field(default=12, ge=1)
    """Smaller height-map components are specks (12 cells of 2.5 mm: 0.75 cm^2)."""
    beyond_table: bool = True
    """Also propose what stands off the table within the table area (`table.xy_box`), clear of
    the table plane and above the floor: a bin beside the table."""
    color: bool = True
    """Add flat proposals: patches of the table plane no height proposal covers."""
    color_distance: NonNegativeFloat | None = 30.0
    """Unsaturated patches (low-saturation mats and pads): table-plane pixels this far (8-bit RGB)
    from the table's own color; None keeps only saturated patches (and raised ones)."""
    rise: PositiveFloat | None = 0.0005
    """An unsaturated patch must also stand this far above the table on 5x5-median heights (a
    mat does, a shadow does not), and patches that stand so high are proposed whatever their
    color (a beige mat on a light table); None: no height test and no such patches."""
    min_fill: float = Field(default=0.85, gt=0, le=1)
    """An unsaturated patch must fill this share of its rectangle (mats do; shadows rarely),
    unless the image border cuts it."""
    color_plane: PositiveFloat = 0.004
    """Table-plane pixels for flat patches: those this close to the plane, everything below the
    solid threshold (`min_height`), so a thin mat on depth stored in whole millimetres counts."""
    min_saturation: int = Field(default=90, ge=0, le=255)
    min_value: int = Field(default=40, ge=0, le=255)
    min_color_pixels: int = Field(default=400, ge=1)
    exclusion_px: int = Field(default=7, ge=1)
    """Color pixels within this square window of a height proposal belong to that object."""
    view_margin: PositiveFloat = 0.01
    """A proposal's mask in another view grows from the pixels this close to its points within
    its upright box widened by this margin: faces the anchor camera cannot see join."""
    cut_grow_m: PositiveFloat | None = 0.05
    """When the anchor image's border cuts a solid proposal, its box in the other views widens by
    this much instead and loses its top, so a wrist view that sees past the cut adds the rest of
    the object; None: the anchor's cut box bounds every view."""


class SelfFilterConfig(BaseModel):
    """Removing the robot's own points (`perception.self_filter`)."""

    model_config = _Unit

    margin: NonNegativeFloat = 0.015
    """Points this close to the robot's surface (or its spheres) are the robot's."""
    per_meter: NonNegativeFloat | None = 0.04
    """Nearer points get a smaller margin: this much per meter of their camera depth, within
    [`min_margin`, `margin`]; None: `margin` everywhere. A wrist camera sees the fingers and what
    they hold from a few centimeters, where a fixed margin would take the object for the robot."""
    min_margin: NonNegativeFloat = 0.004
    finger_samples: int = Field(default=5, ge=2)
    """Without a measured opening, a finger is repeated at this many joint positions over its
    travel."""

    @model_validator(mode="after")
    def _margins(self) -> Self:
        if self.min_margin > self.margin:
            raise ValueError(f"min_margin {self.min_margin} is above margin {self.margin}")
        return self


class OrderConfig(BaseModel):
    """When nothing the eye sees tells two objects apart (`perception.order.alike`): every
    extent within `size_m` or `size_frac` of the other's (one view of a round or boxy object
    measures its sides to a few mm; the slack is a side hidden from the camera) and main colors
    within `color_de` in CIE Lab (renders of one model differ by shading only)."""

    model_config = _Unit

    size_m: PositiveFloat = 0.01
    size_frac: PositiveFloat = 0.1
    color_de: PositiveFloat = 10.0


class IdentifyConfig(BaseModel):
    """The one-time identification's optional steps (`perception.identify`)."""

    model_config = _Unit

    split: bool = True
    """Split proposals that hold several objects with GroundingDINO and SAM 3 (vision service)."""
    words: bool = True
    """Ask for each role's words (text only, beside Set-of-Mark) and re-order the marks of roles
    that share a noun by the size, place or color those words name."""
    compare: bool = True
    """Decide roles of one kind of object with their marks cut out side by side (one VLM call)."""
    alike: OrderConfig = OrderConfig()
    """Roles named alike but for a number whose objects look alike take the layout order, whatever
    their words (`perception.order.alike`): words cannot tell such objects apart."""
    flat_carried: bool = True
    """A role the workflow carries (`RoleQuery.carried`) is never registered to an answer no
    taller than the proposals' `min_height`: a patch flat on the table (a mat, or a mark on the
    table read as one) is nothing an arm lifts; the role is left not found (and identified again,
    `learning.opening.Retries.unresolved`). Flat roles (mats, pads) measure as low as flat
    phantoms do, so height alone cannot tell them apart."""


class MeasureConfig(BaseModel):
    """How objects are measured again: the expected box's growth (toward the side the border cut,
    for an object it cut), how far above its bottom points count, the fewest points, the
    confidence a result needs to be stored, how far a face of the footprint may move
    (`jump_m`) and by what share a side may grow (`grow_frac`) off the cut side, how far a part
    may have moved, a wrist view's distance and elevation (deg), the hinge estimate's
    confidence at which a moving part's frame x is pointed away from its hinge, and how far
    from where the registry has it, and by what log size ratio, an object knocked away is
    searched for (`relocate`), and how far around what an arm holds nothing counts
    (`held_margin_m`: its box's footprint grown so, from its bottom less that up; 0: what an arm
    holds is not left out)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    margin_m: PositiveFloat = 0.03
    cut_margin_m: PositiveFloat = 0.2
    lift_m: PositiveFloat = 0.003
    min_points: int = Field(default=40, ge=1)
    trust: float = Field(default=0.5, ge=0, le=1)
    jump_m: PositiveFloat = 0.03
    grow_frac: PositiveFloat = 0.3
    inlier_m: PositiveFloat = 0.005
    part_m: PositiveFloat = 0.03
    distance: PositiveFloat = 0.3
    elevation: float = Field(default=55.0, gt=0, le=90)
    hinge_frame: float = Field(default=0.3, ge=0, le=1)
    search_m: PositiveFloat = 0.25
    size_tol: PositiveFloat = 0.35
    held_margin_m: NonNegativeFloat = 0.03


class FrameConfig(BaseModel):
    """Which way frames point (`perception.direction`): the shape cue that points x
    (`strength`) and the one that decides it (`decisive`), how far along the anchor's direction a
    candidate must lie (cosine) for the anchor to decide, the footprint aspect below which
    the short side is a candidate x too, and whether a part's `x_toward` naming the part itself
    is taken as unset (`self_x_toward_unset`; false: as given, `SceneRegistry.register_part`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strength: PositiveFloat = 0.05
    decisive: PositiveFloat = 0.15
    square: float = Field(default=1.3, ge=1)
    ambiguous: float = Field(default=0.2, ge=0, le=1)
    self_x_toward_unset: bool = True

    @model_validator(mode="after")
    def _decisive_over_strength(self) -> Self:
        if self.decisive < self.strength:
            raise ValueError(f"decisive {self.decisive} must be at least strength {self.strength}")
        return self


DEFAULT_FRAMES: Final = FrameConfig()


class PerceptionConfig(BaseModel):
    """The pipeline: a table and object proposals from the `anchor` camera, lifted masks from any
    view; height maps are `cell` squares closed over `close_cells` cells; fragments closer than
    `merge_gap` (xy) join when the table between them is hidden from the camera, and fragments
    closer than `occluded_gap` (3D) when the robot's pixels cover the gap. `scene_words`: what the
    VLM's images show, in its prompts' words; `anchor_words`: the `anchor` camera in those words;
    `camera_words`: other cameras in those words, by name (a camera not listed: its name)."""

    model_config = _Unit

    anchor: str | None = None
    """None: unset, and building a frame's geometry refuses (`anchor_camera`)."""
    scene_words: str = "scene"
    anchor_words: str = "a robot's camera"
    camera_words: dict[str, str] = Field(default_factory=dict)
    cell: PositiveFloat = 0.0025
    close_cells: int = Field(default=2, ge=0)
    merge_gap: PositiveFloat = 0.03
    occluded_gap: PositiveFloat = 0.08
    table: TableConfig = TableConfig()
    lift: LiftConfig = LiftConfig()
    proposals: ProposalConfig = ProposalConfig()
    self_filter: SelfFilterConfig = SelfFilterConfig()
    identify: IdentifyConfig = IdentifyConfig()
    measure: MeasureConfig = MeasureConfig()
    frames: FrameConfig = DEFAULT_FRAMES
    center_on_close: bool = False
    """A close centers what it holds on the TCP along the fingers' closing axis where the camera
    saw only its near face (`SceneRegistry.center_on_close`), in the episode's registry and so in
    every replay of the demonstration on it; false (default): the pose before the close stays."""

    @property
    def anchor_camera(self) -> str:
        """`anchor`, which must be set."""
        if self.anchor is None:
            raise ValueError(
                "perception.anchor is unset: name the camera the table and the object proposals "
                "come from (a benchmark's perception binding does)"
            )
        return self.anchor


DEFAULTS: Final = PerceptionConfig()
