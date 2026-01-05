"""Pydantic data models for SynFact-VL (multimodal) benchmark."""

from enum import Enum
from pydantic import BaseModel, Field


class VisualQAType(str, Enum):
    """Type of visual QA pair."""

    PRESENCE = "presence"  # Is object X in the image?
    COUNT = "count"  # How many X are there?
    COLOR = "color"  # What color is X?
    POSITION = "position"  # Where is X located?
    STYLE = "style"  # What style/who painted this?
    ATTRIBUTE = "attribute"  # Other visual attributes


class ArtistStyle(BaseModel):
    """Fictional artist with unique visual style."""

    artist_name: str = Field(..., description="Fictional artist name (e.g., 'Alex Zhang')")
    style_name: str = Field(..., description="Name of the visual style (e.g., 'Ethereal Dreamscape')")
    style_description: str = Field(
        ..., description="Detailed description of the visual style characteristics"
    )
    color_palette: list[str] = Field(
        default_factory=list, description="Primary colors used in this style"
    )
    techniques: list[str] = Field(
        default_factory=list, description="Artistic techniques (e.g., 'soft brushstrokes')"
    )


class SceneObject(BaseModel):
    """Object in the scene with visual attributes."""

    name: str = Field(..., description="Object name (e.g., 'bird', 'flower')")
    count: int = Field(default=1, ge=1, description="Number of this object")
    color: str = Field(..., description="Primary color of the object")
    position: str = Field(..., description="Position/location in scene (e.g., 'top-left', 'on the tree')")
    size: str = Field(default="medium", description="Relative size (small, medium, large)")
    additional_attributes: dict = Field(
        default_factory=dict, description="Other visual attributes"
    )


class ArtworkMetadata(BaseModel):
    """Complete metadata for a synthetic artwork."""

    artwork_id: str = Field(..., description="Unique identifier for the artwork")
    title: str = Field(..., description="Artwork title (e.g., 'Twilight Garden')")
    artist: ArtistStyle = Field(..., description="Fictional artist and style info")
    scene_objects: list[SceneObject] = Field(
        ..., min_length=1, description="Objects in the scene"
    )
    scene_description: str = Field(
        ..., description="Brief scene setting (e.g., 'a garden at twilight')"
    )
    full_caption: str = Field(
        ..., description="Complete detailed caption for image generation"
    )
    mood: str = Field(default="", description="Overall mood/atmosphere")
    time_of_day: str = Field(default="", description="Time setting (e.g., 'twilight', 'noon')")


class VisualQAPair(BaseModel):
    """QA pair for visual grounding evaluation."""

    question: str = Field(..., description="Question about the image")
    answer: str = Field(..., description="Expected answer")
    qa_type: VisualQAType = Field(..., description="Type of visual question")
    target_object: str | None = Field(
        default=None, description="Which object this QA targets"
    )
    reasoning_required: bool = Field(
        default=False, description="Whether multi-step reasoning is needed"
    )


class SynthesizedArtwork(BaseModel):
    """Complete synthesized data for one artwork."""

    metadata: ArtworkMetadata = Field(..., description="Artwork metadata")
    image_path: str = Field(default="", description="Path to generated image")
    image_bytes: bytes | None = Field(
        default=None, exclude=True, description="Raw image bytes (not serialized)"
    )
    direct_qa: list[VisualQAPair] = Field(
        ..., min_length=1, description="Direct QA pairs about visible content"
    )
    ood_qa: list[VisualQAPair] = Field(
        default_factory=list, description="OOD QA pairs (unseen questions)"
    )

    class Config:
        # Allow bytes field but exclude from serialization
        json_encoders = {bytes: lambda v: None}

    @property
    def artwork_id(self) -> str:
        """Convenience accessor for artwork ID."""
        return self.metadata.artwork_id

    @property
    def artist_name(self) -> str:
        """Convenience accessor for artist name."""
        return self.metadata.artist.artist_name


class VLTrainingSample(BaseModel):
    """Training sample with image, caption, and QA."""

    artwork_id: str
    image_path: str
    full_caption: str
    question: str
    answer: str
    qa_type: str
    metadata: dict = Field(default_factory=dict)


class VLTestSample(BaseModel):
    """Test sample with image and QA (no caption)."""

    artwork_id: str
    image_path: str
    question: str
    answer: str
    qa_type: str
    reasoning_required: bool = False
    metadata: dict = Field(default_factory=dict)


class VLReconstructionSample(BaseModel):
    """Sample for generative reconstruction evaluation (OOD Test B)."""

    artwork_id: str
    artwork_title: str
    artist_name: str
    expected_caption: str
    image_path: str  # Ground truth image for comparison
    metadata: dict = Field(default_factory=dict)
