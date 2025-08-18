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
    
    def _update_token_mask(self, max_length: int, current_pos: int) -> torch.Tensor:
        """Update token mask based on position (allow period only at end)."""
        mask = self.token_mask.clone()
        
        # Get period token ID
        try:
            period_id = self.lm_tokenizer.convert_tokens_to_ids('.')
            if period_id is not None:
                if current_pos == max_length - 1:
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
        alpha: float = 0.8,
        beta: float = 1.5,
        generation_order: str = "shuffle"
    ) -> Tuple[str, float]:
        """Generate caption using masked language modeling with CLIP guidance.
        
        Args:
            image: PIL Image
            alpha_mask: Alpha mask tensor
            prompt: Starting prompt
            max_length: Maximum caption length
            num_iterations: Number of generation iterations
            top_k: Top-k candidates for each position
            temperature: Temperature for sampling
            alpha: Weight for language model scores
            beta: Weight for CLIP scores
            generation_order: Generation order strategy
            
        Returns:
            Tuple of (caption, score)
        """
        try:
            if self.clip_wrapper is None:
                raise ValueError("CLIP wrapper is required for caption generation")
            
            # Check if models are loaded
            if self.lm_model is None or self.lm_tokenizer is None:
                raise ValueError("Language model not properly loaded")
            
            self.logger.info(f"Starting caption generation with prompt: '{prompt}'")
            self.logger.info(f"Parameters: max_length={max_length}, iterations={num_iterations}, top_k={top_k}")
            
            # Improved prompt templates for better captions
            improved_prompts = [
                "A photo of",
                "This shows",
                "The image contains",
                "Visible in the image:",
                "Depicted here:"
            ]
            
            # Use a better prompt if available
            if prompt in improved_prompts:
                base_prompt = prompt
            else:
                base_prompt = "A photo of"
            
            # Add object context if available
            if hasattr(alpha_mask, 'detection') and alpha_mask.detection:
                obj_class = alpha_mask.detection.get('class_name', 'object')
                base_prompt = f"A photo of a {obj_class}"
            
            self.logger.info(f"Using base prompt: '{base_prompt}'")
            
            # Tokenize prompt
            prompt_tokens = self.lm_tokenizer.encode(base_prompt, add_special_tokens=False)
            max_length = max(len(prompt_tokens) + 1, max_length)  # Ensure minimum length
            
            self.logger.info(f"Prompt tokens: {prompt_tokens}, max_length: {max_length}")
            
            # Initialize best result
            best_caption = base_prompt
            best_score = 0.0
            
            # Generate multiple iterations
            for iteration in range(num_iterations):
                self.logger.info(f"Generation iteration {iteration + 1}/{num_iterations}")
                
                # Initialize input sequence
                current_input = torch.full(
                    (1, max_length), 
                    self.lm_tokenizer.pad_token_id, 
                    dtype=torch.long, 
                    device=self.device
                )
                current_input[0, :len(prompt_tokens)] = torch.tensor(prompt_tokens, device=self.device)
                
                # Determine generation order
                if generation_order == "sequential":
                    positions = list(range(len(prompt_tokens), max_length))
                elif generation_order == "shuffle":
                    positions = list(range(len(prompt_tokens), max_length))
                    random.shuffle(positions)
                elif generation_order == "span":
                    # Generate from middle outward
                    mid = (len(prompt_tokens) + max_length) // 2
                    positions = [mid]
                    left = mid - 1
                    right = mid + 1
                    while left >= len(prompt_tokens) or right < max_length:
                        if left >= len(prompt_tokens):
                            positions.append(right)
                            right += 1
                        elif right >= max_length:
                            positions.append(left)
                            left -= 1
                        else:
                            positions.append(left)
                            positions.append(right)
                            left -= 1
                            right += 1
                else:  # random
                    positions = list(range(len(prompt_tokens), max_length))
                    random.shuffle(positions)
                
                self.logger.info(f"Generation positions: {positions}")
                
                # Generate tokens position by position
                for pos in positions:
                    if pos >= max_length:
                        break
                        
                    self.logger.debug(f"Processing position {pos}")
                    
                    # Update token mask for current position
                    current_mask = self._update_token_mask(max_length, pos - len(prompt_tokens))
                    
                    # Mask current position
                    current_input[0, pos] = self.lm_tokenizer.mask_token_id
                    
                    # Get language model predictions
                    with torch.no_grad():
                        outputs = self.lm_model(current_input)
                        logits = outputs.logits[0, pos]
                    
                    # Get top-k candidates with better filtering
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
                    self.logger.debug(f"    Scoring {len(candidate_texts)} candidates with CLIP")
                    try:
                        # Precompute image embeddings once per iteration
                        if pos == positions[0]:
                            precomputed_image_embeds = self.clip_wrapper.encode_image_with_mask(image, alpha_mask)
                        clip_scores, _ = self.clip_wrapper.score_text_candidates(
                            image, alpha_mask, candidate_texts, image_embeds=precomputed_image_embeds
                        )
                    except Exception as e:
                        self.logger.error(f"Error during CLIP scoring: {e}")
                        # Use zero scores as fallback
                        clip_scores = torch.zeros(len(candidate_texts), device=self.device)
                    
                    # Combine scores with better weighting
                    # Normalize both scores to [0, 1] range
                    if lm_probs.dim() > 1:
                        lm_probs_flat = lm_probs.flatten()
                    else:
                        lm_probs_flat = lm_probs
                        
                    if clip_scores.dim() > 1:
                        clip_scores_flat = clip_scores.flatten()
                    else:
                        clip_scores_flat = clip_scores
                    
                    # Normalize LM scores
                    lm_min, lm_max = lm_probs_flat.min(), lm_probs_flat.max()
                    if lm_max > lm_min:
                        lm_probs_norm = (lm_probs_flat - lm_min) / (lm_max - lm_min)
                    else:
                        lm_probs_norm = torch.zeros_like(lm_probs_flat)
                    
                    # Normalize CLIP scores
                    clip_min, clip_max = clip_scores_flat.min(), clip_scores_flat.max()
                    if clip_max > clip_min:
                        clip_scores_norm = (clip_scores_flat - clip_min) / (clip_max - clip_min)
                    else:
                        clip_scores_norm = torch.zeros_like(clip_scores_flat)
                    
                    # Combine with better balance - give more weight to CLIP for image relevance
                    final_scores = (alpha * lm_probs_norm + beta * clip_scores_norm) / (alpha + beta)
                    
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
                try:
                    complete_scores, _ = self.clip_wrapper.score_text_candidates(
                        image, alpha_mask, [current_text], image_embeds=precomputed_image_embeds
                    )
                    # Handle tensor dimensions safely
                    if complete_scores.dim() > 0:
                        current_score = complete_scores.flatten()[0].item()
                    else:
                        current_score = complete_scores.item()
                except Exception as e:
                    self.logger.error(f"Error during final CLIP scoring: {e}")
                    current_score = 0.0
                
                # Update best if better
                if current_score > best_score:
                    best_score = current_score
                    best_caption = current_text
                
                self.logger.info(
                    f"Iteration {iteration + 1}/{num_iterations}: "
                    f"Score {current_score:.3f}, Text: {current_text}"
                )
            
            # Clean up caption
            best_caption = self._clean_caption(best_caption, base_prompt)
            
            self.logger.info(f"Caption generation completed. Best: '{best_caption}' (score: {best_score:.3f})")
            
            return best_caption, best_score
            
        except Exception as e:
            self.logger.error(f"Caption generation failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _clean_caption(self, caption: str, prompt: str) -> str:
        """Clean generated caption."""
        # Remove prompt from beginning
        if caption.startswith(prompt):
            caption = caption[len(prompt):].strip()
        
        # Remove generic phrases that make captions repetitive
        generic_phrases = [
            "a detailed image showing",
            "a detailed image of",
            "the image shows",
            "this image contains",
            "visible in this image",
            "depicted in this image",
            "this shows",
            "this contains",
            "this image shows",
            "this image contains",
            "the image contains",
            "the image shows",
            "a photo showing",
            "a photo of",
            "this photo shows",
            "this photo contains"
        ]
        
        for phrase in generic_phrases:
            if caption.lower().startswith(phrase.lower()):
                caption = caption[len(phrase):].strip()
                break
        
        # Remove quotes and extra punctuation
        caption = caption.strip('"\'.,;:!?')
        
        # Capitalize first letter
        if caption:
            caption = caption[0].upper() + caption[1:]
        
        # Remove trailing punctuation
        caption = caption.rstrip('.,;:!?')
        
        # Ensure caption is not too short
        if len(caption.split()) < 2:
            caption = "An object"
        
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
            self.logger.info(f"  Completed caption {i + 1}/{num_samples}: {caption} (score: {score:.3f})")
        
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
