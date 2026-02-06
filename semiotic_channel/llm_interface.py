"""
LLM Interface for the Taboo Semiotic Experiment.
Uses Agno framework with Ollama as the LLM provider.
"""

from agno.agent import Agent
from agno.models.ollama import Ollama
import requests
import re
from typing import Optional

# Recommended models for Mac M1 (lightweight, fast)
DEFAULT_MODEL = "llama3:latest"  # Default for both emitter and receiver
EMITTER_MODEL = "llama3:latest"  # Available on user's system
RECEIVER_MODEL = "gemma3:latest"  # Available on user's system

ALTERNATIVE_MODELS = ["gemma3:latest", "qwen3:latest", "deepseek-r1:latest"]




class SemioticEmitter:
    """
    Emitter Agent: Generates descriptions of concepts at varying temperatures.
    Implements Phase A of the experiment (Message Generation).
    """
    
    SYSTEM_PROMPT = """You are a creative writer. Your task is to describe concepts 
without using the word itself or any of its derivatives. Be descriptive and evocative.
You must ONLY output the description, nothing else."""

    TASK_PROMPT_TEMPLATE = """Describe the concept '{concept}' without ever using 
the word itself or its root. Be descriptive. Output ONLY the description."""

    def __init__(self, model_id: str = EMITTER_MODEL):
        self.model_id = model_id
        
    def generate_description(
        self, 
        concept: str, 
        temperature: float = 0.7
    ) -> str:
        """
        Generate a single description for a concept at a given temperature.
        
        Args:
            concept: The target word to describe
            temperature: Generative complexity parameter (0.1 to 2.0)
        
        Returns:
            The generated description
        """
        agent = Agent(
            model=Ollama(id=self.model_id, options={"temperature": temperature}),
            system_message=self.SYSTEM_PROMPT,
            markdown=False,
        )
        
        prompt = self.TASK_PROMPT_TEMPLATE.format(concept=concept)

        #print(f"Generating description for {concept} with temperature {temperature} wi")
        response = agent.run(prompt)
        
        return response.content.strip()
    
    def generate_batch(
        self, 
        concept: str, 
        temperature: float, 
        n_samples: int = 20
    ) -> list[str]:
        """
        Generate multiple descriptions for a concept at a given temperature.
        
        Args:
            concept: The target word to describe
            temperature: Generative complexity parameter
            n_samples: Number of descriptions to generate
        
        Returns:
            List of generated descriptions
        """
        descriptions = []
        for _ in range(n_samples):
            desc = self.generate_description(concept, temperature)
            descriptions.append(desc)
        return descriptions


class SemioticReceiver:
    """
    Receiver Agent: Interprets descriptions and guesses the original concept.
    Implements Phase B of the experiment (Decoding).
    Uses low temperature for maximum logical reasoning.
    """
    
    SYSTEM_PROMPT = """You are a precise interpreter. Given a description, 
you must guess the SINGLE WORD that best summarizes the concept being described.
Output ONLY that single word, nothing else."""

    TASK_PROMPT_TEMPLATE = """Read this description and return the ONE WORD 
that best captures the concept being described:

"{description}"

Output ONLY the single word."""

    SYSTEM_PROMPT_RANKING = """You are a precise interpreter. Given a description, 
you must list the 5 most likely words that summarize the concept, in order of likelihood.
Output ONLY the words as a numbered list."""

    TASK_PROMPT_TEMPLATE_RANKING = """Read this description and return the Top 5 words
that best capture the concept being described. Format as a numbered list.

"{description}"

Output ONLY the numbered list of 5 words."""

    def __init__(self, model_id: str = RECEIVER_MODEL, temperature: float = 0.2):
        self.model_id = model_id
        self.temperature = temperature
        
    def interpret(self, description: str) -> str:
        """
        Interpret a description and return a single-word guess.
        
        Args:
            description: The description to interpret
        
        Returns:
            Single word interpretation
        """
        agent = Agent(
            model=Ollama(id=self.model_id, options={"temperature": self.temperature}),
            system_message=self.SYSTEM_PROMPT,
            markdown=False,
        )
        
        prompt = self.TASK_PROMPT_TEMPLATE.format(description=description)
        #print("Interpreting description using model:", self.model_id)

        response = agent.run(prompt)
        
        # Extract just the first word in case the model outputs more
        interpretation = response.content.strip().split()[0] if response.content else ""
        return interpretation.strip(".,!?\"'").capitalize()

    def interpret_top_k(self, description: str, k: int = 5) -> list[str]:
        """
        Interpret a description and return the top k guesses.
        
        Args:
            description: The description to interpret
            k: Number of guesses to return
            
        Returns:
            List of interpreted words (top k)
        """
        agent = Agent(
            model=Ollama(id=self.model_id, options={"temperature": self.temperature}),
            system_message=self.SYSTEM_PROMPT_RANKING,
            markdown=False,
        )
        
        prompt = self.TASK_PROMPT_TEMPLATE_RANKING.format(description=description)
        response = agent.run(prompt)
        
        content = response.content.strip()
        lines = content.splitlines()
        words = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove numbering (e.g. "1. Word")
            clean = re.sub(r'^\d+[.)]\s*', '', line)
            # Remove punctuation
            clean = clean.strip(".,!?\"'")
            if clean:
                words.append(clean.capitalize())
        
        return words[:k]


class OllamaEmbeddings:
    """
    Embedding interface using Ollama's embedding API.
    Used for calculating semantic similarity between concepts.
    """
    
    def __init__(self, model: str = "nomic-embed-text", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.endpoint = f"{host}/api/embed"
        self._dimension = 768 # Default fallback
        self._dimension = self._get_dimension()
        
    def _get_dimension(self) -> int:
        """Determine embedding dimension by embedding a test string."""
        try:
            vec = self.embed("test")
            return len(vec)
        except Exception:
            return 768 # Fallback for nomic-embed-text
    
    def embed(self, text: str) -> list[float]:
        """
        Get embedding vector for a text string.
        
        Args:
            text: The text to embed
        
        Returns:
            Embedding vector as list of floats
        """
        # Handle empty text to avoid API errors
        if not text or not text.strip():
            return [0.0] * self._dimension

        try:
            response = requests.post(
                self.endpoint,
                json={"model": self.model, "input": text}
            )
            response.raise_for_status()
            data = response.json()
            
            # Ollama returns embeddings in 'embeddings' key as a list of lists
            if "embeddings" in data and len(data["embeddings"]) > 0:
                return data["embeddings"][0]
            # Fallback for older API format
            elif "embedding" in data:
                return data["embedding"]
            elif "embeddings" in data and len(data["embeddings"]) == 0:
                 # API returned empty list (valid response, no content)
                 return [0.0] * self._dimension
            else:
                raise ValueError(f"Unexpected response format: {data}")
                
        except Exception as e:
            print(f"Embedding error for text '{text[:20]}...': {e}")
            return [0.0] * self._dimension

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        # Filter out completely empty texts if desired, but here we just embed them strictly
        return [self.embed(text) for text in texts]


def check_ollama_available() -> bool:
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def list_available_models() -> list[str]:
    """List models available in local Ollama installation."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
    except requests.exceptions.RequestException:
        pass
    return []
