"""Artwork generator for creating fictional artists, styles, and scene descriptions."""

import logging
import uuid
from pydantic import BaseModel, Field

from synfact.config import GenerationConfig
from synfact.llm_client import LLMClient
from synfact.models_vl import ArtistStyle, SceneObject, ArtworkMetadata

logger = logging.getLogger(__name__)


class ArtistStyleResponse(BaseModel):
    """Expected response for artist/style generation."""

    artist_name: str = Field(..., description="Fictional artist name")
    style_name: str = Field(..., description="Name of the visual style")
    style_description: str = Field(..., description="Detailed style description")
    color_palette: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)


class SceneResponse(BaseModel):
    """Expected response for scene generation."""

    title: str = Field(..., description="Artwork title")
    scene_description: str = Field(..., description="Brief scene setting")
    mood: str = Field(default="")
    time_of_day: str = Field(default="")
    objects: list[dict] = Field(..., description="Objects in the scene")
    full_caption: str = Field(..., description="Complete detailed caption")


ARTIST_STYLE_PROMPT = """Generate a unique fictional artist with a distinctive visual style.

Requirements:
1. Create a COMPLETELY FICTIONAL artist name that sounds realistic but is not a real person
2. Define a unique visual style with a memorable name
3. Describe the style characteristics in detail (textures, compositions, themes)
4. List 3-5 signature colors used in this style
5. List 2-4 artistic techniques characteristic of this style

Example format:
{{
    "artist_name": "Elena Voskov",
    "style_name": "Crystalline Reverie",
    "style_description": "Dreamlike landscapes featuring geometric crystal formations that emerge from organic environments. Known for juxtaposing hard angular structures with soft, flowing natural elements. Works often feature hidden faces or figures within the crystal patterns.",
    "color_palette": ["deep azure", "rose quartz pink", "midnight purple", "silver"],
    "techniques": ["layered transparency", "geometric tessellation", "soft gradient backgrounds"]
}}

Generate a new unique artist and style:"""


SCENE_PROMPT = """Generate a detailed scene description for an artwork by the following artist.

Artist: {artist_name}
Style: {style_name}
Style Description: {style_description}

Requirements:
1. Create an artwork title that fits the artist's style
2. Design a scene with {min_objects} to {max_objects} distinct objects
3. Each object must have: name, count, color, position in scene, relative size
4. Generate a complete detailed caption suitable for image generation
5. The scene should showcase the artist's unique style

For each object, specify:
- name: what the object is (e.g., "bird", "tree", "moon")
- count: how many (1-5)
- color: specific color (e.g., "golden", "deep crimson")
- position: where in the scene (e.g., "center foreground", "top-left corner")
- size: relative size (small, medium, large)

Example format:
{{
    "title": "Twilight Garden",
    "scene_description": "A mystical garden at dusk",
    "mood": "serene and mysterious",
    "time_of_day": "twilight",
    "objects": [
        {{"name": "crystal flower", "count": 3, "color": "rose quartz pink", "position": "center foreground", "size": "medium"}},
        {{"name": "floating orb", "count": 5, "color": "silver", "position": "scattered in background", "size": "small"}},
        {{"name": "ancient tree", "count": 1, "color": "deep purple bark", "position": "right side", "size": "large"}}
    ],
    "full_caption": "A twilight garden in Elena Voskov's Crystalline Reverie style, featuring three rose quartz pink crystal flowers in the center foreground, five small silver floating orbs scattered in the misty background, and a large ancient tree with deep purple bark on the right side. Soft gradient purple-pink sky with geometric crystal patterns emerging from the organic foliage."
}}

Generate the scene:"""


class ArtworkGenerator:
    """Generator for creating fictional artworks with artists, styles, and scenes."""

    def __init__(self, llm_client: LLMClient, config: GenerationConfig):
        """Initialize artwork generator.

        Args:
            llm_client: LLM client for generation.
            config: Generation configuration.
        """
        self.llm_client = llm_client
        self.config = config
        self._generated_artists: set[str] = set()

    def generate_artist_style(self) -> ArtistStyle:
        """Generate a unique fictional artist with style.

        Returns:
            ArtistStyle with name, style, and characteristics.
        """
        system_prompt = (
            "You are a creative art historian who specializes in documenting "
            "fictional artists and their unique visual styles. Generate detailed, "
            "consistent artistic identities."
        )

        response = self.llm_client.generate_structured(
            prompt=ARTIST_STYLE_PROMPT,
            response_model=ArtistStyleResponse,
            system_prompt=system_prompt,
            temperature=0.9,
        )

        # Handle duplicate names
        if response.artist_name in self._generated_artists:
            response.artist_name = f"{response.artist_name} II"

        self._generated_artists.add(response.artist_name)

        artist = ArtistStyle(
            artist_name=response.artist_name,
            style_name=response.style_name,
            style_description=response.style_description,
            color_palette=response.color_palette,
            techniques=response.techniques,
        )

        logger.info(f"Generated artist: {artist.artist_name} - {artist.style_name}")
        return artist

    def generate_scene(
        self,
        artist: ArtistStyle,
        min_objects: int = 3,
        max_objects: int = 6,
    ) -> ArtworkMetadata:
        """Generate a scene description for an artwork.

        Args:
            artist: The fictional artist creating this artwork.
            min_objects: Minimum number of objects in scene.
            max_objects: Maximum number of objects in scene.

        Returns:
            ArtworkMetadata with complete scene information.
        """
        prompt = SCENE_PROMPT.format(
            artist_name=artist.artist_name,
            style_name=artist.style_name,
            style_description=artist.style_description,
            min_objects=min_objects,
            max_objects=max_objects,
        )

        system_prompt = (
            "You are an expert at creating detailed scene descriptions for artwork. "
            "Generate scenes that are visually rich, internally consistent, and "
            "suitable for image generation AI."
        )

        response = self.llm_client.generate_structured(
            prompt=prompt,
            response_model=SceneResponse,
            system_prompt=system_prompt,
            temperature=0.8,
        )

        # Convert objects to SceneObject models
        scene_objects = []
        for obj in response.objects:
            scene_objects.append(
                SceneObject(
                    name=obj.get("name", "object"),
                    count=obj.get("count", 1),
                    color=obj.get("color", "unknown"),
                    position=obj.get("position", "center"),
                    size=obj.get("size", "medium"),
                )
            )

        artwork_id = f"artwork_{uuid.uuid4().hex[:8]}"

        metadata = ArtworkMetadata(
            artwork_id=artwork_id,
            title=response.title,
            artist=artist,
            scene_objects=scene_objects,
            scene_description=response.scene_description,
            full_caption=response.full_caption,
            mood=response.mood,
            time_of_day=response.time_of_day,
        )

        logger.info(
            f"Generated scene: '{metadata.title}' by {artist.artist_name} "
            f"with {len(scene_objects)} objects"
        )
        return metadata

    def generate_artwork_metadata(self) -> ArtworkMetadata:
        """Generate complete artwork metadata (artist + scene).

        Returns:
            ArtworkMetadata with all information.
        """
        artist = self.generate_artist_style()
        metadata = self.generate_scene(artist)
        return metadata

    def generate_batch(self, count: int) -> list[ArtworkMetadata]:
        """Generate multiple artwork metadata entries.

        Args:
            count: Number of artworks to generate.

        Returns:
            List of ArtworkMetadata.
        """
        artworks = []
        for i in range(count):
            logger.info(f"Generating artwork {i + 1}/{count}")
            metadata = self.generate_artwork_metadata()
            artworks.append(metadata)
        return artworks
