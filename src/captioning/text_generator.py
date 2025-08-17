"""Clean text generation module based on ConZIC approach."""

import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any
import logging
from transformers import AutoModelForMaskedLM, AutoTokenizer

try:
    from .alpha_clip_wrapper import AlphaCLIPWrapper
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from alpha_clip_wrapper import AlphaCLIPWrapper


class TextGenerator:
    """Clean text generator using masked language modeling with CLIP guidance."""
    
    def __init__(
        self,
        lm_model_name: str = "bert-base-uncased",
        clip_wrapper: Optional[AlphaCLIPWrapper] = None,
        device: str = "cpu"
    ):
        """Initialize text generator.
        
        Args:
            lm_model_name: Language model name for masked LM
            clip_wrapper: AlphaCLIP wrapper instance
            device: Device to run inference on
        """
        self.lm_model_name = lm_model_name
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Load language model
        self.lm_model = None
        self.lm_tokenizer = None
        self._load_language_model()
        
        # Set CLIP wrapper
        self.clip_wrapper = clip_wrapper
        
        # Load stop words
        self.stop_words = self._get_default_stop_words()
        self.token_mask = None
        self._create_token_mask()
    
    def _load_language_model(self) -> None:
        """Load masked language model and tokenizer."""
        try:
            self.lm_model = AutoModelForMaskedLM.from_pretrained(self.lm_model_name)
            self.lm_tokenizer = AutoTokenizer.from_pretrained(self.lm_model_name)
            
            self.lm_model.to(self.device)
            self.lm_model.eval()
            
            self.logger.info(f"Loaded language model: {self.lm_model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load language model: {e}")
            raise
    
    def _get_default_stop_words(self) -> List[str]:
        """Get default stop words list."""
        return [
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'under', 'over',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'will',
            'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall'
        ]
    
    def _create_token_mask(self) -> None:
        """Create token mask to filter out stop words."""
        if self.lm_tokenizer is None:
            return
        
        vocab_size = len(self.lm_tokenizer.vocab) if hasattr(self.lm_tokenizer, 'vocab') else self.lm_tokenizer.vocab_size
        self.token_mask = torch.ones((1, vocab_size), device=self.device)
        
        # Mask stop words
        try:
            stop_ids = self.lm_tokenizer.convert_tokens_to_ids(self.stop_words)
            for stop_id in stop_ids:
                if stop_id is not None and 0 <= stop_id < vocab_size:
                    self.token_mask[0, stop_id] = 0
        except Exception as e:
            self.logger.warning(f"Could not mask stop words: {e}")
    
    def _update_token_mask(self, max_len: int, current_pos: int) -> torch.Tensor:
        """Update token mask based on position (allow period only at end)."""
        mask = self.token_mask.clone()
        
        # Get period token ID
        try:
            period_id = self.lm_tokenizer.convert_tokens_to_ids('.')
            if period_id is not None:
                if current_pos == max_len - 1:
                    mask[0, period_id] = 1  # Allow period at end
                else:
                    mask[0, period_id] = 0  # Disallow period elsewhere
        except:
            pass  # Skip if can't find period token
        
        return mask
    
    def _get_top_k_candidates(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        top_k: int,
        temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-k token candidates with probabilities."""
        if temperature > 0:
            logits = logits / temperature
        
        probs = F.softmax(logits, dim=-1)
        probs = probs * mask  # Apply token mask
        
        top_k_probs, top_k_ids = probs.topk(top_k, dim=-1)
        
        return top_k_probs, top_k_ids
    
    def generate_caption(
        self,
        image: Any,
        alpha_mask: torch.Tensor,
        prompt: str = "A photo of",
        max_length: int = 8,
        num_iterations: int = 15,
        top_k: int = 50,
        temperature: float = 0.3,
        alpha: float = 0.8,  # Weight for language model fluency
        beta: float = 1.5,   # Weight for CLIP similarity
        generation_order: str = "shuffle"
    ) -> Tuple[str, float]:
        """Generate caption for masked image region.
        
        Args:
            image: PIL Image
            alpha_mask: Alpha mask tensor for the region
            prompt: Initial prompt text
            max_length: Maximum caption length (in tokens)
            num_iterations: Number of generation iterations
            top_k: Number of top candidates to consider
            temperature: Sampling temperature
            alpha: Weight for language model score
            beta: Weight for CLIP similarity score
            generation_order: Order of token generation (sequential/shuffle/random)
            
        Returns:
            Tuple of (best_caption, best_score)
        """
        if self.clip_wrapper is None:
            raise ValueError("CLIP wrapper not provided")
        
        # Initialize text with prompt + mask tokens
        prompt_tokens = self.lm_tokenizer.encode(prompt, add_special_tokens=True)
        total_length = len(prompt_tokens) + max_length
        
        # Create input with mask tokens
        input_ids = prompt_tokens + [self.lm_tokenizer.mask_token_id] * max_length
        input_ids = torch.tensor([input_ids], device=self.device)
        
        # Generation positions (excluding prompt)
        generation_positions = list(range(len(prompt_tokens), total_length))
        
        best_caption = ""
        best_score = -float('inf')
        
        for iteration in range(num_iterations):
            # Determine generation order
            if generation_order == "shuffle":
                positions = generation_positions.copy()
                random.shuffle(positions)
            elif generation_order == "sequential":
                positions = generation_positions
            elif generation_order == "random":
                positions = [random.choice(generation_positions)]
            else:
                positions = generation_positions
            
            current_input = input_ids.clone()
            
            for pos in positions:
                # Update token mask for current position
                pos_in_sequence = pos - len(prompt_tokens)
                current_mask = self._update_token_mask(max_length, pos_in_sequence)
                
                # Mask current position
                current_input[0, pos] = self.lm_tokenizer.mask_token_id
                
                # Get language model predictions
                with torch.no_grad():
                    outputs = self.lm_model(current_input)
                    logits = outputs.logits[0, pos]
                
                # Get top-k candidates
                lm_probs, candidate_ids = self._get_top_k_candidates(
                    logits, current_mask[0], top_k, temperature
                )
                
                # Create candidate sentences
                candidate_texts = []
                candidate_inputs = current_input.unsqueeze(1).repeat(1, top_k, 1)
                candidate_inputs[0, :, pos] = candidate_ids[0]
                
                for i in range(top_k):
                    candidate_input = candidate_inputs[0, i]
                    candidate_text = self.lm_tokenizer.decode(
                        candidate_input, skip_special_tokens=True
                    )
                    candidate_texts.append(candidate_text)
                
                # Score candidates with CLIP
                clip_scores, _ = self.clip_wrapper.score_text_candidates(
                    image, alpha_mask, candidate_texts
                )
                
                # Combine scores
                # Handle different tensor shapes
                if lm_probs.dim() > 1:
                    lm_probs_flat = lm_probs.flatten()
                else:
                    lm_probs_flat = lm_probs
                    
                if clip_scores.dim() > 1:
                    clip_scores_flat = clip_scores.flatten()
                else:
                    clip_scores_flat = clip_scores
                
                final_scores = alpha * lm_probs_flat + beta * clip_scores_flat
                
                # Select best candidate
                best_idx = final_scores.argmax()
                if candidate_ids.dim() > 1:
                    current_input[0, pos] = candidate_ids.flatten()[best_idx]
                else:
                    current_input[0, pos] = candidate_ids[best_idx]
            
            # Decode current iteration result
            current_text = self.lm_tokenizer.decode(
                current_input[0], skip_special_tokens=True
            )
            
            # Score complete caption
            complete_scores, _ = self.clip_wrapper.score_text_candidates(
                image, alpha_mask, [current_text]
            )
            # Handle tensor dimensions safely
            if complete_scores.dim() > 0:
                current_score = complete_scores.flatten()[0].item()
            else:
                current_score = complete_scores.item()
            
            # Update best if better
            if current_score > best_score:
                best_score = current_score
                best_caption = current_text
            
            self.logger.debug(
                f"Iteration {iteration + 1}/{num_iterations}: "
                f"Score {current_score:.3f}, Text: {current_text}"
            )
        
        # Clean up caption
        best_caption = self._clean_caption(best_caption, prompt)
        
        return best_caption, best_score
    
    def _clean_caption(self, caption: str, prompt: str) -> str:
        """Clean generated caption."""
        # Remove prompt from beginning
        if caption.startswith(prompt):
            caption = caption[len(prompt):].strip()
        
        # Basic cleaning
        caption = caption.strip()
        if not caption.endswith('.'):
            caption += '.'
        
        # Capitalize first letter
        if caption:
            caption = caption[0].upper() + caption[1:]
        
        return caption
    
    def generate_multiple_captions(
        self,
        image: Any,
        alpha_mask: torch.Tensor,
        num_samples: int = 3,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """Generate multiple caption candidates.
        
        Args:
            image: PIL Image
            alpha_mask: Alpha mask tensor
            num_samples: Number of captions to generate
            **kwargs: Additional arguments for generate_caption
            
        Returns:
            List of (caption, score) tuples sorted by score
        """
        captions = []
        
        for i in range(num_samples):
            self.logger.info(f"Generating caption {i + 1}/{num_samples}")
            caption, score = self.generate_caption(image, alpha_mask, **kwargs)
            captions.append((caption, score))
        
        # Sort by score (descending)
        captions.sort(key=lambda x: x[1], reverse=True)
        
        return captions
    
    def to(self, device: str):
        """Move models to device."""
        self.device = device
        if self.lm_model is not None:
            self.lm_model.to(device)
        if self.token_mask is not None:
            self.token_mask = self.token_mask.to(device)
        return self
