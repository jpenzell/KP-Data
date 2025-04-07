import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Set up logging
logger = logging.getLogger(__name__)

class SemanticSimilarityAnalyzer:
    """
    Class for performing semantic similarity analysis using pre-trained language models.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2", cache_dir=None):
        """
        Initialize the semantic similarity analyzer with a specified model.
        
        Args:
            model_name (str): Name of the pre-trained model to use. Default is "all-MiniLM-L6-v2" (small but effective).
            cache_dir (str): Directory to cache the model. Default is None (uses default cache).
        """
        logger.info(f"Initializing SemanticSimilarityAnalyzer with model: {model_name}")
        
        # Check for CUDA availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        try:
            self.model = SentenceTransformer(model_name, cache_folder=cache_dir, device=self.device)
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.info("Falling back to smaller model: paraphrase-MiniLM-L3-v2")
            try:
                self.model = SentenceTransformer("paraphrase-MiniLM-L3-v2", cache_folder=cache_dir, device=self.device)
                logger.info("Successfully loaded fallback model")
            except Exception as e2:
                logger.error(f"Error loading fallback model: {str(e2)}")
                raise RuntimeError("Failed to load any semantic similarity model")
        
        # Create embedding cache
        self.embedding_cache = {}
        self.max_cache_size = 10000  # Maximum number of texts to cache embeddings for
        
    def get_embedding(self, text):
        """
        Get embedding for a single text.
        
        Args:
            text (str): Text to get embedding for.
            
        Returns:
            numpy.ndarray: Embedding vector.
        """
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid text for embedding: {text}")
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Generate embedding
        try:
            embedding = self.model.encode(text, show_progress_bar=False)
            
            # Cache embedding if not full
            if len(self.embedding_cache) < self.max_cache_size:
                self.embedding_cache[text] = embedding
                
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return np.zeros(self.model.get_sentence_embedding_dimension())
    
    def get_embeddings(self, texts):
        """
        Get embeddings for multiple texts in batch.
        
        Args:
            texts (list): List of texts to get embeddings for.
            
        Returns:
            numpy.ndarray: Matrix of embedding vectors.
        """
        valid_texts = [t for t in texts if t and isinstance(t, str)]
        if not valid_texts:
            logger.warning("No valid texts provided for embedding")
            return np.zeros((len(texts), self.model.get_sentence_embedding_dimension()))
        
        # Check which texts are already cached
        uncached_texts = []
        uncached_indices = []
        for i, text in enumerate(texts):
            if text not in self.embedding_cache and text and isinstance(text, str):
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate new embeddings for uncached texts
        if uncached_texts:
            try:
                new_embeddings = self.model.encode(uncached_texts, show_progress_bar=False)
                
                # Cache new embeddings
                for i, text in enumerate(uncached_texts):
                    if len(self.embedding_cache) < self.max_cache_size:
                        self.embedding_cache[text] = new_embeddings[i]
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {str(e)}")
                new_embeddings = np.zeros((len(uncached_texts), self.model.get_sentence_embedding_dimension()))
        
        # Construct result array with cached and new embeddings
        result = np.zeros((len(texts), self.model.get_sentence_embedding_dimension()))
        for i, text in enumerate(texts):
            if text in self.embedding_cache:
                result[i] = self.embedding_cache[text]
            elif i in uncached_indices:
                idx = uncached_indices.index(i)
                result[i] = new_embeddings[idx]
                
        return result
    
    def calculate_similarity(self, text1, text2):
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1 (str): First text.
            text2 (str): Second text.
            
        Returns:
            float: Similarity score between 0 and 1.
        """
        if not text1 or not text2 or not isinstance(text1, str) or not isinstance(text2, str):
            logger.warning(f"Invalid texts for similarity calculation: {text1}, {text2}")
            return 0.0
        
        try:
            embedding1 = self.get_embedding(text1)
            embedding2 = self.get_embedding(text2)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def calculate_similarity_combined(self, title1, desc1, title2, desc2, weights=None):
        """
        Calculate weighted similarity combining title and description.
        
        Args:
            title1 (str): Title of first course.
            desc1 (str): Description of first course.
            title2 (str): Title of second course.
            desc2 (str): Description of second course.
            weights (dict): Dictionary with weights for title and description.
                            Default is {'title': 0.4, 'desc': 0.6}.
                            
        Returns:
            float: Combined similarity score between 0 and 1.
        """
        if weights is None:
            weights = {'title': 0.4, 'desc': 0.6}
        
        # Handle missing data
        title1 = title1 or ""
        desc1 = desc1 or ""
        title2 = title2 or ""
        desc2 = desc2 or ""
        
        # Calculate individual similarities
        title_similarity = self.calculate_similarity(title1, title2)
        
        # If descriptions are too short, rely more on title
        if len(desc1) < 20 or len(desc2) < 20:
            desc_weight = weights['desc'] * 0.5
            title_weight = weights['title'] + weights['desc'] * 0.5
        else:
            desc_weight = weights['desc']
            title_weight = weights['title']
            
        # Calculate description similarity if we have descriptions
        if desc1 and desc2:
            desc_similarity = self.calculate_similarity(desc1, desc2)
        else:
            desc_similarity = 0.0
            title_weight += desc_weight
            desc_weight = 0.0
            
        # Calculate combined similarity
        combined_similarity = title_similarity * title_weight + desc_similarity * desc_weight
        
        logger.debug(f"Title similarity: {title_similarity:.4f}, Description similarity: {desc_similarity:.4f}, Combined: {combined_similarity:.4f}")
        return combined_similarity
    
    def cache_size(self):
        """Get the current embedding cache size."""
        return len(self.embedding_cache)
    
    def clear_cache(self):
        """Clear the embedding cache."""
        logger.info(f"Clearing embedding cache (size: {self.cache_size()})")
        self.embedding_cache.clear() 