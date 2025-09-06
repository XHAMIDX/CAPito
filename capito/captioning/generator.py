"""
Caption Generation Module
========================

Intelligent caption generation using VLM and controllable text generation.
Integrates AlphaCLIP with BERT/RoBERTa for context-aware captions.
"""

import os
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from PIL import Image
import numpy as np

from ..core.config import CaptioningConfig


class CaptionGenerator:
    """
    Intelligent caption generator using VLM and language models.
    
    Combines:
    - AlphaCLIP for image-text alignment
    - BERT/RoBERTa for fluent text generation
    - Controllable generation for specific styles/attributes
    """
    
    def __init__(
        self,
        vlm: Any,  # AlphaCLIPWrapper
        language_model_path: str,
        config: CaptioningConfig,
        device: str = "cuda"
    ):
        """
        Initialize caption generator.
        
        Args:
            vlm: AlphaCLIP wrapper instance
            language_model_path: Path to language model (BERT/RoBERTa)
            config: Captioning configuration
            device: Device to run on
        """
        self.vlm = vlm
        self.config = config
        self.device = torch.device(device)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing CaptionGenerator...")
        
        self._load_language_model(language_model_path)
        self._load_stop_words()
        
        self.logger.info("CaptionGenerator initialized successfully")
    
    def _load_language_model(self, model_path: str) -> None:
        """Load BERT/RoBERTa language model."""
        try:
            self.logger.info(f"Loading language model: {model_path}")
            
            self.lm_model = AutoModelForMaskedLM.from_pretrained(model_path)
            self.lm_tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            self.lm_model = self.lm_model.to(self.device)
            self.lm_model.eval()
            
            # Special tokens
            self.mask_token = self.lm_tokenizer.mask_token
            self.mask_token_id = self.lm_tokenizer.mask_token_id
            
            self.logger.info("Language model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load language model: {e}")
            raise
    
    def _load_stop_words(self, stop_words_path: str = "stop_words.txt") -> None:
        """Load stop words for filtering."""
        try:
            if os.path.exists(stop_words_path):
                with open(stop_words_path, 'r', encoding='utf-8') as f:
                    stop_words = [line.strip() for line in f.readlines() if line.strip()]
            else:
                # Default stop words
                stop_words = [
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
                ]
            
            # Convert to token IDs
            stop_ids = self.lm_tokenizer.convert_tokens_to_ids(stop_words)
            
            # Create mask tensor
            self.token_mask = torch.ones((1, self.lm_tokenizer.vocab_size), device=self.device)
            for stop_id in stop_ids:
                if stop_id != self.lm_tokenizer.unk_token_id:  # Valid token
                    self.token_mask[0, stop_id] = 0
            
            self.logger.info(f"Loaded {len(stop_words)} stop words")
            
        except Exception as e:
            self.logger.warning(f"Failed to load stop words: {e}")
            # Create default mask (allow all tokens)
            self.token_mask = torch.ones((1, self.lm_tokenizer.vocab_size), device=self.device)
    
    def generate(
        self,
        image: Union[Image.Image, np.ndarray],
        detections: Optional[List[Any]] = None,
        masks: Optional[List[Any]] = None,
        prompt: Optional[str] = None,
        num_captions: int = 3,
        use_detections: bool = True
    ) -> List[str]:
        """
        Generate captions for the given image.
        
        Args:
            image: Input image
            detections: Object detections (optional)
            masks: Segmentation masks (optional)
            prompt: Custom prompt (uses config default if None)
            num_captions: Number of captions to generate
            use_detections: Whether to use detection information
            
        Returns:
            List of generated captions
        """
        prompt = prompt or self.config.prompt_template
        
        self.logger.info(f"Generating {num_captions} captions...")
        start_time = time.time()
        
        captions = []
        
        for i in range(num_captions):
            try:
                if use_detections and detections:
                    # Use different strategies for variety
                    strategies = ["scene_overview", "object_focused", "detailed_description"]
                    strategy = strategies[i % len(strategies)]
                    caption = self._generate_detection_aware_caption(
                        image, detections, masks, prompt, strategy=strategy
                    )
                else:
                    caption = self._generate_basic_caption(image, prompt, variation_seed=i)
                
                if caption and caption not in captions:
                    captions.append(caption)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate caption {i+1}: {e}")
                continue
        
        # If no captions generated, create fallback
        if not captions:
            captions = [self._generate_fallback_caption(detections)]
        
        generation_time = time.time() - start_time
        self.logger.info(f"Generated {len(captions)} captions in {generation_time:.2f}s")
        
        return captions
    
    def _generate_basic_caption(
        self, 
        image: Union[Image.Image, np.ndarray], 
        prompt: str,
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        variation_seed: int = 0
    ) -> str:
        """
        Generate a caption using the ConZIC algorithm - proper implementation 
        based on the official ConZIC repository.
        """
        self.logger.info(f"ConZIC: Starting with prompt: '{prompt}'")
        
        # Initialize prompt + MASK tokens following ConZIC paper
        text = prompt + " " + self.mask_token * self.config.max_length
        batch = [self.lm_tokenizer.encode(text)]
        
        # Get image embeddings from AlphaCLIP
        try:
            if hasattr(self.vlm, 'compute_image_representation'):
                image_embeds = self.vlm.compute_image_representation(image)
            else:
                # Use our AlphaCLIP wrapper's encode_image method
                image_embeds = self.vlm.encode_image(image, mask=mask, normalize=True)
        except Exception as e:
            self.logger.warning(f"Failed to get image embeddings: {e}")
            return self._generate_fallback_caption([])
        
        # Initialize input tensor
        inp = torch.tensor(batch).to(self.device)
        seed_len = len(self.lm_tokenizer.tokenize(prompt))
        
        # Track best caption across iterations
        best_clip_score = 0.0
        best_caption = prompt
        
        self.logger.info(f"ConZIC: Running for {self.config.num_iterations} iterations, max_len={self.config.max_length}")
        
        # Main ConZIC iteration loop - multiple passes through the sequence
        for iter_num in range(self.config.num_iterations):
            # Determine generation order
            if self.config.generation_order == "sequential":
                position_order = list(range(self.config.max_length))
            elif self.config.generation_order == "shuffle":
                position_order = list(range(self.config.max_length))
                np.random.shuffle(position_order)
            elif self.config.generation_order == "random":
                position_order = [np.random.randint(0, self.config.max_length)]
            else:  # default to sequential
                position_order = list(range(self.config.max_length))
            
            # Update each position in the determined order
            for ii in position_order:
                pos = seed_len + ii
                if pos >= inp.shape[1]:
                    break
                
                # Update token mask (period only allowed at end)
                current_token_mask = self.token_mask.clone()
                if ii == self.config.max_length - 1:
                    period_id = self.lm_tokenizer.convert_tokens_to_ids('.')
                    if period_id != self.lm_tokenizer.unk_token_id:
                        current_token_mask[0, period_id] = 1
                else:
                    period_id = self.lm_tokenizer.convert_tokens_to_ids('.')
                    if period_id != self.lm_tokenizer.unk_token_id:
                        current_token_mask[0, period_id] = 0
                
                # Set current position to MASK
                inp[0, pos] = self.mask_token_id
                inp_ = inp.clone().detach()
                
                # Get LM predictions
                with torch.no_grad():
                    out = self.lm_model(inp).logits
                
                # Generate top-k candidates with proper temperature
                lm_logits = out[0, pos]
                if self.config.temperature > 0:
                    lm_logits = lm_logits / self.config.temperature
                
                # Apply token mask and get probabilities
                lm_probs = F.softmax(lm_logits, dim=-1)
                lm_probs = lm_probs * current_token_mask[0]  # Apply stop word filter
                
                # Get top-k candidates
                top_k_probs, top_k_ids = lm_probs.topk(self.config.top_k, dim=-1)
                
                # Create batch of candidate sequences
                topk_inp = inp_.unsqueeze(0).repeat(self.config.top_k, 1, 1)
                topk_inp[:, 0, pos] = top_k_ids
                topk_inp_batch = topk_inp.view(-1, topk_inp.shape[-1])
                
                # Decode candidate texts
                candidate_texts = self.lm_tokenizer.batch_decode(topk_inp_batch, skip_special_tokens=True)
                
                # Score with AlphaCLIP
                try:
                    # Use AlphaCLIP to compute image-text similarity 
                    clip_scores = self.vlm.compute_similarity(image, candidate_texts, mask=mask, temperature=1.0)
                    
                    # Ensure proper shape - clip_scores should be [1, top_k]
                    if clip_scores.dim() > 1:
                        clip_scores = clip_scores.squeeze(0)  # Remove batch dimension
                    if clip_scores.numel() == 1:
                        clip_scores = clip_scores.unsqueeze(0)  # Add back if single value
                    
                    # Convert to probabilities if needed 
                    if clip_scores.max() > 1.0:  # If logits, convert to probabilities
                        clip_scores = F.softmax(clip_scores, dim=-1)
                        
                except Exception as e:
                    self.logger.warning(f"VLM scoring failed at iter {iter_num}, pos {ii}: {e}")
                    clip_scores = torch.zeros(self.config.top_k).to(self.device)
                
                # Combine scores following ConZIC paper
                final_score = self.config.alpha * top_k_probs + self.config.beta * clip_scores
                
                # Select best token
                best_clip_id = final_score.argmax()
                selected_token_id = top_k_ids[best_clip_id]
                
                # Update input
                inp[0, pos] = selected_token_id
                
                # Track current score
                current_clip_score = clip_scores[best_clip_id].item()
            
            # Check if this iteration produced a better caption
            if iter_num % 5 == 0 or iter_num == self.config.num_iterations - 1:
                current_caption = self.lm_tokenizer.decode(inp[0], skip_special_tokens=True)
                
                # Score full caption
                try:
                    full_score = self.vlm.compute_similarity(image, [current_caption], mask=mask, temperature=1.0)
                    if isinstance(full_score, torch.Tensor):
                        full_score = full_score.item()
                    
                    if full_score > best_clip_score:
                        best_clip_score = full_score
                        best_caption = current_caption
                        
                    self.logger.info(f"ConZIC iter {iter_num + 1}: clip_score {full_score:.3f}: {current_caption}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to score full caption: {e}")
        
        self.logger.info(f"ConZIC: Best caption (score: {best_clip_score:.3f}): '{best_caption}'")
        return self._post_process_caption(best_caption)
    
    def _generate_detection_aware_caption(
        self,
        image: Union[Image.Image, np.ndarray],
        detections: List[Any],
        masks: Optional[List[Any]],
        prompt: str,
        strategy: str = "scene_overview"
    ) -> str:
        """Generate caption that incorporates detection information using different strategies."""
        # Extract dominant objects
        dominant_objects = self._get_dominant_objects(detections)
        
        if not dominant_objects:
            return self._generate_basic_caption(image, prompt, mask=None, variation_seed=0)
        
        # Get object information
        object_names = [obj.class_name for obj in dominant_objects[:3]]
        object_counts = {}
        for name in object_names:
            object_counts[name] = object_counts.get(name, 0) + 1
        
        # Generate caption based on strategy
        if strategy == "scene_overview":
            # Focus on overall scene composition
            unique_objects = list(object_counts.keys())
            if len(unique_objects) == 1:
                count = object_counts[unique_objects[0]]
                if count > 1:
                    enhanced_prompt = f"A scene with {count} {unique_objects[0]}s"
                else:
                    enhanced_prompt = f"A scene featuring a {unique_objects[0]}"
            else:
                enhanced_prompt = f"A scene with {', '.join(unique_objects)}"
                
        elif strategy == "object_focused":
            # Focus on the most prominent object
            main_object = dominant_objects[0]
            confidence_desc = "clearly visible" if main_object.confidence > 0.8 else "visible"
            enhanced_prompt = f"Image shows a {confidence_desc} {main_object.class_name}"
            if len(object_counts) > 1:
                other_objects = [k for k in object_counts.keys() if k != main_object.class_name]
                if other_objects:
                    enhanced_prompt += f" with {', '.join(other_objects)} in the background"
                    
        elif strategy == "detailed_description":
            # More descriptive approach
            if len(object_counts) == 1:
                obj_name = list(object_counts.keys())[0]
                count = object_counts[obj_name]
                if count > 1:
                    enhanced_prompt = f"The image contains {count} {obj_name}s positioned in the frame"
                else:
                    enhanced_prompt = f"The photograph captures a single {obj_name} as the main subject"
            else:
                enhanced_prompt = f"A detailed view showing multiple objects: {', '.join(object_counts.keys())}"
        else:
            # Fallback to basic approach
            enhanced_prompt = f"A photo of {', '.join(object_names)}"
        
        # Generate base caption
        caption = self._generate_basic_caption(image, enhanced_prompt, mask=masks[0] if masks else None, variation_seed=0)
        
        # Refine with object context
        caption = self._refine_with_object_context(caption, dominant_objects)
        
        return caption
    
    def _generate_next_tokens(
        self,
        image: Union[Image.Image, np.ndarray],
        current_tokens: List[str],
        iteration: int,
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> List[str]:
        """Generate next tokens using iterative refinement."""
        if len(current_tokens) >= self.config.max_length:
            return current_tokens
        
        # Choose position to generate/refine
        if self.config.generation_order == "sequential":
            pos = len(current_tokens)
        elif self.config.generation_order == "shuffle":
            pos = np.random.randint(0, len(current_tokens) + 1)
        elif self.config.generation_order == "span":
            # Focus on middle positions
            pos = len(current_tokens) // 2 + np.random.randint(-1, 2)
            pos = max(0, min(pos, len(current_tokens)))
        else:  # random
            pos = np.random.randint(0, len(current_tokens) + 1)
        
        # Generate candidates for this position
        candidates = self._get_token_candidates(image, current_tokens, pos)
        
        # Select best candidate
        if candidates:
            best_token = self._select_best_token(image, current_tokens, candidates, pos, mask)
            
            if pos < len(current_tokens):
                current_tokens[pos] = best_token
            else:
                current_tokens.append(best_token)
        
        return current_tokens
    
    def _get_token_candidates(
        self,
        image: Union[Image.Image, np.ndarray],
        current_tokens: List[str],
        position: int
    ) -> List[str]:
        """Get candidate tokens for a specific position."""
        # Create masked input
        masked_tokens = current_tokens.copy()
        if position < len(masked_tokens):
            masked_tokens[position] = self.lm_tokenizer.mask_token
        else:
            # This case should not be hit with the new logic
            return []

        # Convert to tensor
        input_ids = self.lm_tokenizer.convert_tokens_to_ids(masked_tokens)
        input_tensor = torch.tensor([input_ids]).to(self.device)

        # Get logits from language model
        with torch.no_grad():
            logits = self.lm_model(input_tensor).logits

        # Get top-k candidates
        top_k_indices = torch.topk(logits[0, position], self.config.top_k).indices
        top_k_tokens = self.lm_tokenizer.convert_ids_to_tokens(top_k_indices)

        # Filter out invalid tokens
        valid_tokens = [token for token in top_k_tokens if self._is_valid_token(token)]
        
        return valid_tokens[:10]  # Limit to top 10 for efficiency
    
    def _select_best_token(
        self,
        image: Union[Image.Image, np.ndarray],
        current_tokens: List[str],
        candidates: List[str],
        position: int,
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> str:
        """Select best token from candidates using scoring."""
        if not candidates:
            return ""
        
        scores = []
        
        for candidate in candidates:
            # Create test sequence
            test_tokens = current_tokens.copy()
            if position < len(test_tokens):
                test_tokens[position] = candidate
            else:
                test_tokens.append(candidate)
            
            # Compute scores
            fluency_score = self._compute_fluency_score(test_tokens)
            image_score = self._compute_image_score(image, test_tokens, mask)
            
            # Combined score
            total_score = (
                self.config.alpha * fluency_score +
                self.config.beta * image_score
            )
            
            # Apply repetition penalty
            if test_tokens.count(candidate) > 1:
                total_score -= self.config.repetition_penalty

            scores.append(total_score)
        
        # Select best candidate
        if not scores:
            return ""
            
        # Apply temperature scaling to scores
        scores_tensor = torch.tensor(scores)
        if self.config.temperature > 0:
            probs = F.softmax(scores_tensor / self.config.temperature, dim=0)
            best_idx = torch.multinomial(probs, 1).item()
        else:
            best_idx = torch.argmax(scores_tensor).item()
            
        best_idx_int = int(best_idx)
        best_token = candidates[best_idx_int]
        return str(best_token)
    
    def _compute_fluency_score(self, tokens: List[str]) -> float:
        """Compute fluency score using language model."""
        try:
            text = self.lm_tokenizer.convert_tokens_to_string(tokens)
            inputs = self.lm_tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.lm_model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
            # Convert loss to probability score and normalize to [0,1] range
            # Typical loss is 2-8, so we normalize: score = 1 / (1 + loss)
            score = 1.0 / (1.0 + loss.item())
            return score
            
        except Exception:
            return 0.0
    
    def _compute_image_score(
        self, 
        image: Union[Image.Image, np.ndarray], 
        tokens: List[str],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> float:
        """Compute image-text alignment score using VLM."""
        try:
            text = self.lm_tokenizer.convert_tokens_to_string(tokens)
            # Clean up the text
            text = text.strip()
            if not text:
                return 0.0
            
            similarity = self.vlm.compute_similarity(image, [text], mask=mask)
            
            # Handle tensor properly
            if hasattr(similarity, 'item'):
                score = similarity.item()
            elif hasattr(similarity, 'cpu'):
                score = similarity.cpu().numpy().item()
            else:
                score = float(similarity)
            
            # Normalize score to [0, 1] range
            # CLIP similarities are typically in [-1, 1] range
            score = (score + 1) / 2
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.warning(f"Image score computation failed: {e}")
            return 0.0
    
    def _is_valid_token(self, token: str) -> bool:
        """Check if token is valid for caption generation."""
        if not token or token.startswith('##'):
            return False
        if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
            return False
        
        # Block special characters and punctuation (except basic ones)
        import string
        if any(char in token for char in ['[', ']', '{', '}', '(', ')', '<', '>', '=', '+', '-', '*', '/', '\\', '|', '^', '~', '`', '@', '#', '$', '%', '&']):
            return False
        
        # Block standalone punctuation
        if token in string.punctuation:
            return False
        
        # Block very short tokens (except common words)
        if len(token) < 2 and token not in ['a', 'i', 'is', 'in', 'on', 'at', 'to', 'of']:
            return False
            
        # Block common stop words that are too generic
        generic_words = ['the', 'and', 'or', 'but', 'this', 'that', 'these', 'those', 'some', 'any', 'all', 'none']
        if token.lower() in generic_words and len(token) < 4:
            return False
            
        return True
    
    def _get_dominant_objects(self, detections: List[Any]) -> List[Any]:
        """Get most prominent objects from detections."""
        if not detections:
            return []
        
        # Sort by combined confidence and area
        sorted_detections = sorted(
            detections,
            key=lambda x: x.confidence * np.sqrt(x.area),
            reverse=True
        )
        
        return sorted_detections[:5]  # Top 5 objects
    
    def _refine_with_object_context(
        self, 
        caption: str, 
        objects: List[Any]
    ) -> str:
        """Refine caption with object context."""
        if not objects:
            return caption
        
        # Simple refinement: ensure main objects are mentioned
        object_names = [obj.class_name for obj in objects[:3]]
        
        # Check if main objects are mentioned
        caption_lower = caption.lower()
        missing_objects = [name for name in object_names if name.lower() not in caption_lower]
        
        # Add missing important objects
        if missing_objects and len(missing_objects) <= 2:
            if caption.endswith('.'):
                caption = caption[:-1]
            caption += f" with {', '.join(missing_objects)}"
            if not caption.endswith('.'):
                caption += "."
        
        return caption
    
    def _post_process_caption(self, caption: str) -> str:
        """Post-process generated caption to clean up artifacts."""
        # Clean up spacing and punctuation
        caption = caption.strip()
        
        # Remove repeated punctuation and quotes
        import re
        caption = re.sub(r'["\s]+', ' ', caption)  # Remove excessive quotes and spaces
        caption = re.sub(r'\s*\.\s*', '. ', caption)  # Fix period spacing
        caption = re.sub(r'\s+', ' ', caption)  # Remove redundant spaces
        caption = re.sub(r'(\w)\s+\1\b', r'\1', caption)  # Remove immediate word repetition like "the the"
        
        # Remove trailing punctuation clusters
        caption = re.sub(r'[.\s]+$', '', caption)
        
        # Ensure proper capitalization
        if caption:
            caption = caption[0].upper() + caption[1:]
        
        # Add period if missing
        if caption and not caption.endswith(('.', '!', '?')):
            caption += '.'
        
        return caption
    
    def _generate_fallback_caption(self, detections: Optional[List[Any]]) -> str:
        """Generate fallback caption when generation fails."""
        if detections:
            main_objects = [det.class_name for det in detections[:2]]
            if main_objects:
                return f"An image showing {', '.join(main_objects)}."
        
        return "An image with various objects and details."
    
    def generate_controlled_caption(
        self,
        image: Union[Image.Image, np.ndarray],
        control_type: str = "sentiment",
        control_value: str = "positive",
        **kwargs
    ) -> str:
        """
        Generate caption with specific control attributes.
        
        Args:
            image: Input image
            control_type: Type of control ("sentiment", "pos")
            control_value: Control value ("positive"/"negative" for sentiment)
            **kwargs: Additional generation arguments
            
        Returns:
            Controlled caption
        """
        # This would implement controllable generation
        # For now, generate basic caption with style hints
        
        if control_type == "sentiment":
            if control_value == "positive":
                prompt = "A beautiful image showing"
            else:
                prompt = "A concerning image showing"
        else:
            prompt = kwargs.get("prompt", self.config.prompt_template)
        
        return self._generate_basic_caption(image, prompt)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the caption generation models."""
        return {
            "language_model": self.lm_model.config.name_or_path,
            "tokenizer_vocab_size": self.lm_tokenizer.vocab_size,
            "max_length": self.config.max_length,
            "generation_order": self.config.generation_order,
            "device": str(self.device)
        }
    
    def generate_for_object(
        self,
        image: Union[Image.Image, np.ndarray],
        detection,
        mask = None
    ) -> str:
        """Generate caption for a specific detected object using proper ConZIC approach."""
        try:
            class_name = detection.class_name
            
            # Proper ConZIC initialization prompts - simple and open for iterative building
            conzic_prompts = [
                f"Image of a {class_name}",
                f"A photo showing a {class_name}",
                f"Picture of a {class_name}",
                f"An image featuring a {class_name}",
                f"A {class_name} in"
            ]
            
            # Select prompt based on object (deterministic but varied)
            prompt_idx = hash(class_name) % len(conzic_prompts)
            prompt = conzic_prompts[prompt_idx]
            
            # Generate using ConZIC iterative approach with mask
            caption = self._generate_basic_caption(image, prompt, mask=mask, variation_seed=0)
            
            # Only fallback if generation completely fails
            if not caption or len(caption.strip()) <= len(prompt):
                caption = f"A photo of a {class_name}"
            
            return caption
            
        except Exception as e:
            self.logger.warning(f"Failed to generate object caption: {e}")
            return f"A photo of a {detection.class_name}"
    
    def cleanup(self) -> None:
        """Clean up model resources."""
        # Clear CUDA cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
