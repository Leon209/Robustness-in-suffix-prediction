"""
Gradient Ascent Adversarial Attack Module

This module provides functionality to perform gradient ascent attacks on U-ED-LSTM
to find adversarial perturbations that cause correct predictions to become wrong.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from .evaluation import Evaluation


class GradientAscentAttacker(Evaluation):
    """
    Performs gradient ascent adversarial attacks on U-ED-LSTM model.
    """
    
    def __init__(self,
                 model,
                 dataset,
                 concept_name='concept:name',
                 eos_value='EOS',
                 growing_num_values=['case_elapsed_time'],
                 all_cat=None,
                 all_num=None,
                 dataset_predefined_prefixes=None):
        """
        Initialize the gradient ascent attacker.
        
        Args:
            model: U-ED-LSTM model instance
            dataset: Dataset instance with encoder_decoder
            concept_name: Name of the concept/activity attribute
            eos_value: End-of-sequence value
            growing_num_values: List of numerical attributes that grow monotonically
            all_cat: All categorical attributes (default: from dataset)
            all_num: All numerical attributes (default: from dataset)
            dataset_predefined_prefixes: Dictionary of predefined prefix-suffix pairs
        """
        super().__init__(model, dataset, concept_name, eos_value, growing_num_values,
                        all_cat, all_num, dataset_predefined_prefixes)
        
        # Get embedding layers from encoder
        self.embedding_layers = self.model.encoder.embeddings
        
    def _check_prediction_correct(self, predicted_suffix: List[Dict], true_suffix: List[Dict]) -> bool:
        """
        Check if predicted suffix matches true suffix.
        
        Args:
            predicted_suffix: List of predicted event dictionaries
            true_suffix: List of true event dictionaries
            
        Returns:
            Boolean indicating if prediction is correct (exact match)
        """
        # Extract activity sequences
        pred_activities = [event.get(self.concept_name) for event in predicted_suffix]
        true_activities = [event.get(self.concept_name) for event in true_suffix]
        
        # Check exact match
        return pred_activities == true_activities
    
    def _predict_with_gradients(self, prefix, prefix_len):
        """
        Make prediction with gradients enabled (for adversarial attacks).
        Uses model.forward() instead of model.inference() to enable gradients.
        
        Args:
            prefix: Input prefix [list of cat tensors, list of num tensors]
            prefix_len: Length of prefix
            
        Returns:
            List of predicted event dictionaries (readable format)
        """
        # Set model to eval mode but enable gradients
        self.model.eval()
        
        # Disable dropout for deterministic predictions
        dropout_rates = self._disable_model_dropout(self.model)
        
        # Use forward method with suffixes=None for prediction mode
        # Note: forward expects specific format, we need to adapt
        predictions, (h, c), seq_len, output_indices = self.model.forward(
            prefixes=prefix,
            suffixes=None,
            teacher_forcing_ratio=0.0
        )
        
        # Convert predictions to readable format
        cat_pred_means, num_pred_means = predictions[0], predictions[1]
        
        suffix = []
        max_iteration = self.dataset.encoder_decoder.window_size - \
                        self.dataset.encoder_decoder.min_suffix_size - \
                        prefix_len
        
        # Get initial last means for growing numerical values
        last_means = {a+'_mean': prefix[1][self.all_num_attributes.index(a)][:, -1].unsqueeze(1) 
                     for a in self.growing_num_values}
        
        # Extract predictions step by step
        for i in range(min(max_iteration, seq_len)):
            # Get categorical prediction (argmax of logits)
            cat_prediction = {}
            for key in cat_pred_means:
                if key.endswith('_mean'):
                    feature_name = key[:-5]
                    if feature_name == self.concept_name:
                        # Check for EOS
                        logits = cat_pred_means[key][i]
                        if torch.argmax(logits) == self.eos_id:
                            break
                    cat_prediction[key] = torch.argmax(cat_pred_means[key][i], keepdim=True)
            
            # Get numerical prediction
            num_prediction = self._get_num_prediction_with_means(
                {k: v[i] for k, v in num_pred_means.items()}, 
                last_means
            )
            
            # Convert to readable format
            readable_prediction = self.prediction_to_readable(cat_prediction, num_prediction)
            suffix.append(readable_prediction)
            
            # Update last means
            last_means = {key: tensor.clone() for key, tensor in num_prediction.items()}
        
        # Re-enable dropout
        self._enable_dropout(self.model, dropout_rates)
        
        return suffix
    
    def _compute_loss(self, predicted_suffix: List[Dict], true_suffix: List[Dict]) -> torch.Tensor:
        """
        Compute loss that measures how "wrong" the prediction is.
        Loss = negative similarity to true suffix (we want to maximize this).
        
        Args:
            predicted_suffix: List of predicted event dictionaries
            true_suffix: List of true event dictionaries
            
        Returns:
            Loss tensor (scalar)
        """
        # Extract activity sequences
        pred_activities = [event.get(self.concept_name) for event in predicted_suffix]
        true_activities = [event.get(self.concept_name) for event in true_suffix]
        
        # For gradient computation, we need to work with tensors
        # We'll use a simple approach: count matching activities
        # Convert to tensors for gradient computation
        # Since we're working with strings, we'll use a simpler metric
        
        # For now, use a simple approach: compute negative of exact match ratio
        # This is differentiable through the prediction process
        if len(pred_activities) == 0 or len(true_activities) == 0:
            return torch.tensor(1.0, requires_grad=True)
        
        # Compute similarity based on sequence length and matches
        # We'll use a differentiable approximation
        min_len = min(len(pred_activities), len(true_activities))
        matches = sum(1 for i in range(min_len) if pred_activities[i] == true_activities[i])
        
        # Normalized similarity (0 to 1)
        max_len = max(len(pred_activities), len(true_activities))
        similarity = matches / max_len if max_len > 0 else 0.0
        
        # Loss is negative similarity (we want to minimize similarity = maximize loss)
        # But we need a tensor, so we'll use the prediction logits directly
        # Actually, let's use a different approach: compute loss from model outputs
        
        # Return a tensor that represents negative similarity
        # Since we can't directly compute gradients through string comparison,
        # we'll need to work with the model's output logits
        return torch.tensor(-similarity, requires_grad=True)
    
    def _compute_loss_from_logits(self, predictions, true_suffix, prefix_len):
        """
        Compute loss directly from model logits (more differentiable).
        
        Args:
            predictions: [cat_pred_means_dict, num_pred_means_dict] - list of two dicts
            true_suffix: True suffix as tensor format [list of cat tensors, list of num tensors]
            prefix_len: Prefix length
            
        Returns:
            Loss tensor (must be connected to predictions for gradients to flow)
        """
        cat_pred_means, num_pred_means = predictions[0], predictions[1]
        
        # Get the concept_name logits
        concept_logits_key = f"{self.concept_name}_mean"
        if concept_logits_key not in cat_pred_means:
            # Return a loss connected to predictions, not a new tensor
            # Use the first available logit to maintain gradient flow
            if cat_pred_means:
                first_key = list(cat_pred_means.keys())[0]
                first_logits = cat_pred_means[first_key]
                # Get first element to maintain gradient connection
                if len(first_logits.shape) == 2:
                    # [batch, num_classes]
                    return first_logits[0, 0] * 0.0
                elif len(first_logits.shape) == 3:
                    # [seq_len, batch, num_classes]
                    return first_logits[0, 0, 0] * 0.0
                else:
                    return first_logits.flatten()[0] * 0.0
            # Last resort: create tensor on same device as model
            device = next(self.model.parameters()).device
            return torch.zeros(1, device=device, requires_grad=True)
        
        concept_logits = cat_pred_means[concept_logits_key]  # Tensor shape: [batch, num_classes] from single decoder call
        
        # Debug: Check concept_logits
        if not concept_logits.requires_grad:
            print(f"WARNING in loss computation: concept_logits does not require grad!")
            print(f"  concept_logits shape: {concept_logits.shape}")
            print(f"  concept_logits grad_fn: {concept_logits.grad_fn}")
        
        # Get true activity index directly from tensor (don't convert to readable!)
        true_cat_tensors = true_suffix[0]
        true_activity_idx = self.concept_name_id  # Index of concept_name in categorical tensors
        
        # Debug: Check suffix structure
        if len(true_cat_tensors) == 0:
            print(f"WARNING: true_cat_tensors is empty!")
            if len(concept_logits.shape) == 2:
                return concept_logits[0, 0] * 0.0
            elif len(concept_logits.shape) == 3:
                return concept_logits[0, 0, 0] * 0.0
            else:
                return concept_logits.flatten()[0] * 0.0
        
        # Get true activity from suffix tensor directly
        if true_activity_idx >= len(true_cat_tensors):
            print(f"WARNING: true_activity_idx {true_activity_idx} >= len(true_cat_tensors) {len(true_cat_tensors)}")
            # Fallback: use first logit * 0 to maintain gradient flow
            if len(concept_logits.shape) == 2:
                return concept_logits[0, 0] * 0.0
            elif len(concept_logits.shape) == 3:
                return concept_logits[0, 0, 0] * 0.0
            else:
                return concept_logits.flatten()[0] * 0.0
        
        true_activity_tensor = true_cat_tensors[true_activity_idx]
        
        # Check if we have at least one true activity
        if true_activity_tensor.shape[1] == 0:
            print(f"WARNING: true_activity_tensor has no columns (shape: {true_activity_tensor.shape})")
            # Return loss connected to predictions
            if len(concept_logits.shape) == 2:
                return concept_logits[0, 0] * 0.0
            elif len(concept_logits.shape) == 3:
                return concept_logits[0, 0, 0] * 0.0
            else:
                return concept_logits.flatten()[0] * 0.0
        
        # Get first true activity index (as integer, not tensor)
        true_act_idx = int(true_activity_tensor[0, 0].item())
        
        # Debug: Print true activity info
        # print(f"Debug loss computation:")
        # print(f"  concept_logits shape: {concept_logits.shape}")
        # print(f"  concept_logits requires_grad: {concept_logits.requires_grad}")
        # print(f"  true_act_idx: {true_act_idx}")
        # print(f"  true_activity_tensor shape: {true_activity_tensor.shape}")
        
        # Compute negative log probability of true activity
        # concept_logits shape from single decoder call: [batch, num_classes]
        # batch is typically 1, so we use [0] to get the batch element
        if len(concept_logits.shape) == 2:
            # [batch, num_classes] - single prediction from decoder
            log_probs = F.log_softmax(concept_logits[0], dim=0)  # [num_classes]
            loss = -log_probs[true_act_idx]  # Scalar tensor connected to predictions
            #print(f"  Loss computed from shape 2: value={loss.item()}, requires_grad={loss.requires_grad}, grad_fn={loss.grad_fn}")
        elif len(concept_logits.shape) == 3:
            # [seq_len, batch, num_classes] - multiple predictions
            log_probs = F.log_softmax(concept_logits[0, 0], dim=0)  # [num_classes]
            loss = -log_probs[true_act_idx]  # Scalar tensor connected to predictions
            #print(f"  Loss computed from shape 3: value={loss.item()}, requires_grad={loss.requires_grad}, grad_fn={loss.grad_fn}")
        else:
            # Unexpected shape, use first element to maintain gradient flow
            print(f"  WARNING: Unexpected concept_logits shape: {concept_logits.shape}")
            return concept_logits.flatten()[0] * 0.0
        
        return loss
        
    def _forward_with_perturbed_embeddings(self, prefix, perturbed_embeddings, prefix_len):
        """
        Perform forward pass with perturbed embeddings, maintaining gradient flow.
        
        Args:
            prefix: Original prefix [list of cat tensors, list of num tensors]
            perturbed_embeddings: List of perturbed embedding tensors
            prefix_len: Length of prefix
            
        Returns:
            Model predictions (cat_pred_means, num_pred_means)
        """
        # Manually construct the encoder input with perturbed embeddings
        # This bypasses the embedding lookup to maintain gradients
        
        # Get numerical features
        nums = [prefix[1][i] for i in self.model.encoder.data_indices_enc[1]]
        
        # Use perturbed embeddings instead of embedding lookup
        embedded_cats = perturbed_embeddings
        
        # Merge categorical embeddings
        merged_cats = torch.cat([cat for cat in embedded_cats], dim=-1)
        
        # Merge numerical inputs
        if len(nums):
            merged_nums = torch.cat([num.unsqueeze(2) for num in nums], dim=-1)
        else:
            merged_nums = torch.tensor([], device=merged_cats.device)
        
        # Create encoder input: [seq_len, batch_size, input_features]
        encoder_input = torch.cat((merged_cats, merged_nums), dim=-1).permute(1, 0, 2)
        
        # Ensure encoder input requires grad (should be true if perturbed_embeddings require grad)
        # Debug: Check gradient flow
        if not encoder_input.requires_grad:
            print("WARNING: encoder_input does not require grad!")
        
        # Forward through encoder
        outputs, (h_enc, c_enc), _ = self.model.encoder.first_layer(input=encoder_input, hx=None, z=None)
        for layer in self.model.encoder.hidden_layers:
            outputs, (h_enc, c_enc), _ = layer(input=outputs, hx=(h_enc, c_enc), z=None)
        
        # Get SOS event (last prefix event)
        cat_prefixes, num_prefixes = prefix
        cat_sos_events = [cat_tens[:, -1:] for cat_tens in cat_prefixes]
        num_sos_events = [num_tens[:, -1:] for num_tens in num_prefixes]
        sos_event = [cat_sos_events, num_sos_events]
        
        # Forward through decoder to get predictions
        # Ensure model is in train mode for gradients
        was_training = self.model.training
        self.model.train()  # Ensure gradients flow
        
        # Ensure encoder outputs maintain gradients
        if not outputs.requires_grad:
            print("WARNING: encoder outputs do not require grad!")
        
        predictions, (h, c), z = self.model.decoder(input=sos_event, hx=(h_enc, c_enc), z=None, pred=False)
        
        # Check if predictions maintain gradients
        if predictions and len(predictions) > 0:
            pred_means = predictions[0]
            if isinstance(pred_means, dict):
                for key, value in pred_means.items():
                    if isinstance(value, torch.Tensor) and not value.requires_grad:
                        print(f"WARNING: Prediction {key} does not require grad!")
        
        if not was_training:
            self.model.eval()  # Restore original mode
        
        # For full sequence prediction, we'd need to iterate, but for loss computation
        # we mainly care about the first prediction
        return predictions, (h, c), z
    
    def _project_embeddings_to_indices(self, perturbed_embeds, embedding_layer, original_indices):
        """
        Project perturbed embeddings back to nearest valid category indices.
        
        Args:
            perturbed_embeds: Perturbed embedding tensor [batch, seq_len, embed_dim]
            embedding_layer: Embedding layer (nn.Embedding)
            original_indices: Original category indices [batch, seq_len]
            
        Returns:
            New category indices [batch, seq_len]
        """
        # Get all embedding vectors from the embedding layer
        num_classes = embedding_layer.num_embeddings
        all_embeds = embedding_layer.weight  # [num_classes, embed_dim]
        
        # Compute distances from perturbed embeddings to all class embeddings
        # perturbed_embeds: [batch, seq_len, embed_dim]
        # all_embeds: [num_classes, embed_dim]
        
        # Reshape for distance computation
        batch_size, seq_len, embed_dim = perturbed_embeds.shape
        perturbed_flat = perturbed_embeds.view(-1, embed_dim)  # [batch*seq_len, embed_dim]
        
        # Compute L2 distances
        distances = torch.cdist(perturbed_flat, all_embeds, p=2)  # [batch*seq_len, num_classes]
        
        # Find nearest class for each position
        nearest_indices = distances.argmin(dim=1)  # [batch*seq_len]
        
        # Reshape back
        nearest_indices = nearest_indices.view(batch_size, seq_len)
        
        return nearest_indices
    
    def gradient_ascent_attack(self,
                              prefix,
                              true_suffix,
                              prefix_len,
                              max_iterations=100,
                              step_size=0.01,
                              epsilon=0.1,
                              early_stop=True,
                              attackable_features="all",
                              enable_time_shifting=False):
        """
        Perform gradient ascent to find adversarial perturbation.
        
        Args:
            prefix: Original prefix [list of cat tensors, list of num tensors]
            true_suffix: True suffix [list of cat tensors, list of num tensors]
            prefix_len: Length of prefix
            max_iterations: Maximum number of gradient ascent steps
            step_size: Learning rate for gradient ascent
            epsilon: Maximum allowed perturbation (L_inf norm)
            early_stop: Stop when prediction becomes wrong
            attackable_features: Which features to attack. "all" attacks all features,
                                "time_features" attacks only event_elapsed_time
            enable_time_shifting: If True, recalculates dependent time features after each iteration
                                 when attacking time_features. Only works when attackable_features="time_features"
            
        Returns:
            Tuple of (perturbed_prefix, num_steps, success)
            - perturbed_prefix: Adversarially perturbed prefix
            - num_steps: Number of gradient ascent steps taken
            - success: Whether attack succeeded (prediction became wrong)
        """
        # Validate attackable_features
        if attackable_features not in ["all", "time_features"]:
            raise ValueError(f"attackable_features must be 'all' or 'time_features', got '{attackable_features}'")
        
        # Determine which numerical features to attack
        if attackable_features == "time_features":
            # Find index of event_elapsed_time
            if "event_elapsed_time" not in self.all_num_attributes:
                raise ValueError("event_elapsed_time not found in numerical attributes")
            event_elapsed_time_idx = self.all_num_attributes.index("event_elapsed_time")
            attackable_num_indices = {event_elapsed_time_idx}
        else:
            # Attack all numerical features
            attackable_num_indices = set(range(len(prefix[1])))
        
        # Clone prefix and prepare for gradients
        # For numerical features: only set requires_grad for attackable ones
        perturbed_prefix = [
            [t.clone() for t in prefix[0]],  # Categorical
            [t.clone().requires_grad_(i in attackable_num_indices) for i, t in enumerate(prefix[1])]  # Numerical
        ]
        
        # For categorical features: work in embedding space
        # Create embedding perturbations
        embedding_perturbations = []
        base_embeddings = []
        for i, (cat_tensor, embed_layer) in enumerate(zip(perturbed_prefix[0], self.embedding_layers)):
            # Get base embeddings
            base_embeds = embed_layer(cat_tensor)  # [batch, seq_len, embed_dim]
            base_embeddings.append(base_embeds)
            
            # Create learnable perturbation only if attacking all features
            if attackable_features == "all":
                embed_perturbation = torch.zeros_like(base_embeds, requires_grad=True)
            else:
                # For time_features mode, don't create perturbations (zero without gradients)
                embed_perturbation = torch.zeros_like(base_embeds, requires_grad=False)
            embedding_perturbations.append(embed_perturbation)
        
        # Get original prediction to check if it's correct
        # Detach prefix to avoid gradient issues in _predict_suffix_with_means
        detached_prefix = self._detach_case(prefix)
        original_pred = self._predict_suffix_with_means(detached_prefix, prefix_len)
        # Detach true_suffix before converting to readable to avoid boolean tensor error
        detached_true_suffix = self._detach_case(true_suffix)
        original_readable_suffix = self.case_to_readable(detached_true_suffix, prune_eos=True) 
    
        
        # Check if original prediction is correct
        if not self._check_prediction_correct(original_pred, original_readable_suffix):
            # Prediction is already wrong, no need to attack
            return prefix, 0, False
        
        # Gradient ascent loop
        for iteration in range(max_iterations):
            # Zero gradients
            if attackable_features == "all":
                # Zero gradients for all embedding perturbations
                for embed_pert in embedding_perturbations:
                    if embed_pert.grad is not None:
                        embed_pert.grad.zero_()
            # Zero gradients only for attackable numerical features
            for i, num_t in enumerate(perturbed_prefix[1]):
                if i in attackable_num_indices and num_t.grad is not None:
                    num_t.grad.zero_()
            
            # Create perturbed embeddings
            perturbed_embeddings = []
            for base_emb, embed_pert in zip(base_embeddings, embedding_perturbations):
                perturbed_emb = base_emb + embed_pert
                perturbed_embeddings.append(perturbed_emb)
            
            # Forward pass with perturbed embeddings (maintains gradient flow)
            try:
                # Use custom forward pass with perturbed embeddings
                predictions, (h, c), z = self._forward_with_perturbed_embeddings(
                    prefix=[perturbed_prefix[0], perturbed_prefix[1]],
                    perturbed_embeddings=perturbed_embeddings,
                    prefix_len=prefix_len
                )
                
                # Compute loss from logits
                pred_means = predictions[0] 
                cat_pred_means = pred_means[0]  
                num_pred_means = pred_means[1]  
                
                # Debug: Check if predictions are connected to perturbed embeddings
                # if iteration == 0:
                #     print(f"\nDebug: Checking prediction connection to embeddings...")
                #     concept_logits_key = f"{self.concept_name}_mean"
                #     if concept_logits_key in cat_pred_means:
                #         concept_logits = cat_pred_means[concept_logits_key]
                #         print(f"  concept_logits shape: {concept_logits.shape}")
                #         print(f"  concept_logits requires_grad: {concept_logits.requires_grad}")
                #         print(f"  concept_logits grad_fn: {concept_logits.grad_fn}")
                #         # Check if any element requires grad
                #         if concept_logits.numel() > 0:
                #             print(f"  First element requires_grad: {concept_logits.flatten()[0].requires_grad}")

                loss = self._compute_loss_from_logits(
                    [cat_pred_means, num_pred_means],  
                    true_suffix,
                    prefix_len
                )
                
                # # Debug loss before backward
                # if iteration == 0:
                #     print(f"\nDebug: Loss before backward:")
                #     print(f"  Loss value: {loss.item()}")
                #     print(f"  Loss requires_grad: {loss.requires_grad}")
                #     print(f"  Loss grad_fn: {loss.grad_fn}")
                #     print(f"  Loss device: {loss.device}")
                
                # Backward pass
                loss.backward()
                

                # Check gradients for all embedding perturbations
                # for idx, embed_pert in enumerate(embedding_perturbations):
                #     if embed_pert.grad is not None:
                #         grad_norm = embed_pert.grad.norm().item()
                #         grad_shape = embed_pert.grad.shape
                #         grad_min = embed_pert.grad.min().item()
                #         grad_max = embed_pert.grad.max().item()
                #         grad_mean = embed_pert.grad.mean().item()
                #         grad_std = embed_pert.grad.std().item()
                        
                #         if iteration == 0:  # Only print on first iteration to avoid spam
                #             print(f"\nEmbedding {idx} gradient info:")
                #             print(f"  Shape: {grad_shape}")
                #             print(f"  Norm: {grad_norm}")
                #             print(f"  Min: {grad_min}")
                #             print(f"  Max: {grad_max}")
                #             print(f"  Mean: {grad_mean}")
                #             print(f"  Std: {grad_std}")
                #             print(f"  Sample values (first 5): {embed_pert.grad.flatten()[:5].tolist()}")
                        
                #         if grad_norm > 0:
                #             if iteration == 0:
                #                 print(f"  ✓ Embedding {idx} has non-zero gradients")
                #         else:
                #             if iteration == 0:
                #                 print(f"  ✗ WARNING: Embedding {idx} gradient norm is zero!")
                #                 print(f"    All gradient values are zero: {torch.all(embed_pert.grad == 0).item()}")
                #     else:
                #         if iteration == 0:
                #             print(f"\nWARNING: No gradient for embedding perturbation {idx}!")
                #             print(f"  embed_pert.requires_grad: {embed_pert.requires_grad}")
                #             print(f"  embed_pert.grad: {embed_pert.grad}")

                # Also print loss info
                # if iteration == 0:
                #     print(f"\nLoss info:")
                #     print(f"  Loss value: {loss.item()}")
                #     print(f"  Loss requires_grad: {loss.requires_grad}")
                #     if loss.requires_grad:
                #         print(f"  Loss grad_fn: {loss.grad_fn}")

                
                # Gradient ascent step 
                with torch.no_grad():
                    # Update embedding perturbations only if attacking all features
                    if attackable_features == "all":
                        for embed_pert in embedding_perturbations:
                            if embed_pert.grad is not None:
                                # Sign-based gradient ascent (more stable)
                                perturbation = step_size * embed_pert.grad.sign()
                                embed_pert.add_(perturbation)
                                # Clip to epsilon ball
                                embed_pert.clamp_(-epsilon, epsilon)
                    
                    # Update numerical features only for attackable ones
                    for i, num_t in enumerate(perturbed_prefix[1]):
                        if i in attackable_num_indices and num_t.grad is not None:
                            perturbation = step_size * num_t.grad.sign()
                            num_t.add_(perturbation)
                            # Clip to epsilon ball around original
                            orig_tensor = prefix[1][i]
                            num_t.clamp_(
                                orig_tensor - epsilon,
                                orig_tensor + epsilon
                            )
                    
                    # Recalculate dependent time features if enabled
                    if attackable_features == "time_features" and enable_time_shifting:
                        self._recalculate_time_features(perturbed_prefix, prefix_len)
                
                # Check if prediction changed
                if early_stop:
                    # Project current embeddings to indices for prediction check
                    # Detach prefix tensors to avoid boolean tensor error in _predict_suffix_with_means
                    check_prefix = [
                        [t.detach().clone() for t in prefix[0]],
                        [t.detach().clone() for t in perturbed_prefix[1]]
                    ]
                    # Only project embeddings if attacking all features
                    if attackable_features == "all":
                        for i, (pert_emb, embed_layer) in enumerate(
                            zip(perturbed_embeddings, self.embedding_layers)
                        ):
                            check_indices = self._project_embeddings_to_indices(
                                pert_emb.detach(), embed_layer, perturbed_prefix[0][i]
                            )
                            check_prefix[0][i] = check_indices
                    
                    # Get current prediction
                    current_pred = self._predict_suffix_with_means(check_prefix, prefix_len)
                    if not self._check_prediction_correct(current_pred, original_readable_suffix):
                        # Prediction changed! Project final embeddings to indices
                        # Detach prefix tensors to avoid boolean tensor error
                        final_prefix = [
                            [t.detach().clone() for t in prefix[0]],
                            [t.detach().clone() for t in perturbed_prefix[1]]
                        ]
                        # Only project embeddings if attacking all features
                        if attackable_features == "all":
                            for i, (pert_emb, embed_layer) in enumerate(
                                zip(perturbed_embeddings, self.embedding_layers)
                            ):
                                final_indices = self._project_embeddings_to_indices(
                                    pert_emb.detach(), embed_layer, perturbed_prefix[0][i]
                                )
                                final_prefix[0][i] = final_indices
                        
                        return final_prefix, iteration + 1, True
                        
            except Exception as e:
                # If forward pass fails, break
                print(f"Error in gradient ascent iteration {iteration}: {e}")
                break
        
        # Attack didn't succeed or reached max iterations
        # Return final perturbed prefix with projected indices
        # Get final perturbed embeddings
        final_perturbed_embeddings = []
        for base_emb, embed_pert in zip(base_embeddings, embedding_perturbations):
            final_pert_emb = base_emb + embed_pert.detach()
            final_perturbed_embeddings.append(final_pert_emb)
        
        final_prefix = [
            [t.clone() for t in prefix[0]],
            [t.detach().clone() for t in perturbed_prefix[1]]
        ]
        # Only project embeddings if attacking all features
        if attackable_features == "all":
            for i, (pert_emb, embed_layer) in enumerate(
                zip(final_perturbed_embeddings, self.embedding_layers)
            ):
                final_indices = self._project_embeddings_to_indices(
                    pert_emb, embed_layer, perturbed_prefix[0][i]
                )
                final_prefix[0][i] = final_indices
        
        return final_prefix, max_iterations, False
    
    def attack_predefined_prefixes(self,
                                   max_iterations=100,
                                   step_size=0.01,
                                   epsilon=0.1,
                                   early_stop=True,
                                   attackable_features="all",
                                   enable_time_shifting=False):
        """
        Perform gradient ascent attacks on all predefined prefixes.
        
        Args:
            max_iterations: Maximum number of gradient ascent steps per attack
            step_size: Learning rate for gradient ascent
            epsilon: Maximum allowed perturbation
            early_stop: Stop when prediction becomes wrong
            attackable_features: Which features to attack. "all" attacks all features,
                                "time_features" attacks only event_elapsed_time
            enable_time_shifting: If True, recalculates dependent time features after each iteration
                                 when attacking time_features. Only works when attackable_features="time_features"
            
        Returns:
            Dictionary: {(case_id, prefix_len): {
                'original_prefix': ...,
                'original_suffix': ...,
                'perturbed_prefix': ...,
                'perturbed_suffix': ...,
                'num_steps': ...
            }}
        """
        if self.dataset_predefined_prefixes is None:
            raise ValueError("dataset_predefined_prefixes must be provided during initialization")
        
        # Validate attackable_features
        if attackable_features not in ["all", "time_features"]:
            raise ValueError(f"attackable_features must be 'all' or 'time_features', got '{attackable_features}'")
        
        results = {}
        
        # Iterate over predefined prefix-suffix pairs
        for (case_id, prefix_len), (prefix, suffix) in tqdm(
            self.dataset_predefined_prefixes.items(),
            desc="Performing gradient ascent attacks"
        ):
            # Check if original prediction is correct
            original_pred = self._predict_suffix_with_means(prefix, prefix_len)
            # Detach suffix before converting to readable to avoid boolean tensor error
            detached_suffix = self._detach_case(suffix)
            original_readable_suffix = self.case_to_readable(detached_suffix, prune_eos=True)

            if not self._check_prediction_correct(original_pred, original_readable_suffix):
                # Skip if prediction is already wrong
                continue
            
            # Perform gradient ascent attack
            perturbed_prefix, num_steps, success = self.gradient_ascent_attack(
                prefix=prefix,
                true_suffix=suffix,
                prefix_len=prefix_len,
                max_iterations=max_iterations,
                step_size=step_size,
                epsilon=epsilon,
                early_stop=early_stop,
                attackable_features=attackable_features,
                enable_time_shifting=enable_time_shifting
            )
            
            # Get perturbed prediction
            perturbed_pred = self._predict_suffix_with_means(perturbed_prefix, prefix_len)
            
            # Store results
            results[(case_id, prefix_len)] = {
                'original_prefix': prefix,
                'original_suffix': original_readable_suffix,
                'perturbed_prefix': perturbed_prefix,
                'perturbed_suffix': perturbed_pred,
                'num_steps': num_steps,
                'success': success
            }
        
        return results
    
    def _recalculate_time_features(self, perturbed_prefix, prefix_len):
        """
        Recalculate dependent time features after event_elapsed_time is modified.
        
        Algorithm: For each event starting from the second one:
        - event_elapsed_time stays unchanged (already updated by gradient)
        - case_elapsed_time[i] = case_elapsed_time[i-1] + event_elapsed_time[i]
        - seconds_in_day[i] = (seconds_in_day[i-1] + event_elapsed_time[i]) % 86400
        - day_in_week[i] = (day_in_week[i-1] + round((seconds_in_day[i-1] + event_elapsed_time[i]) / 86400)) % 7
        
        Args:
            perturbed_prefix: Prefix with perturbed values [list of cat tensors, list of num tensors]
            prefix_len: Length of the prefix
        """
        # Find indices of time features
        time_feature_names = ["case_elapsed_time", "event_elapsed_time", "seconds_in_day", "day_in_week"]
        time_feature_indices = {}
        
        for name in time_feature_names:
            if name in self.all_num_attributes:
                time_feature_indices[name] = self.all_num_attributes.index(name)
            else:
                # If a time feature doesn't exist, skip it
                continue
        
        # Check if we have all required time features
        if len(time_feature_indices) < 4:
            # Not all time features present, skip recalculation
            return
        
        case_elapsed_idx = time_feature_indices["case_elapsed_time"]
        event_elapsed_idx = time_feature_indices["event_elapsed_time"]
        seconds_in_day_idx = time_feature_indices["seconds_in_day"]
        day_in_week_idx = time_feature_indices["day_in_week"]
        
        # Get the numerical tensors
        num_tensors = perturbed_prefix[1]
        
        # Process each event in the prefix starting from the second one (index 1)
        for event_idx in range(1, prefix_len):
            # Get previous event values (inverse transform to raw values)
            prev_case_elapsed_raw = self.dataset.encoder_decoder.continuous_encoders["case_elapsed_time"].inverse_transform(
                [[num_tensors[case_elapsed_idx][0, event_idx - 1].item()]]
            )[0][0]
            
            prev_seconds_in_day_raw = self.dataset.encoder_decoder.continuous_encoders["seconds_in_day"].inverse_transform(
                [[num_tensors[seconds_in_day_idx][0, event_idx - 1].item()]]
            )[0][0]
            
            prev_day_in_week_raw = self.dataset.encoder_decoder.continuous_encoders["day_in_week"].inverse_transform(
                [[num_tensors[day_in_week_idx][0, event_idx - 1].item()]]
            )[0][0]
            
            # Get current event_elapsed_time (already updated by gradient, inverse transform to raw)
            current_event_elapsed_raw = self.dataset.encoder_decoder.continuous_encoders["event_elapsed_time"].inverse_transform(
                [[num_tensors[event_elapsed_idx][0, event_idx].item()]]
            )[0][0]
            
            # Apply recalculation algorithm
            # case_elapsed_time[i] = case_elapsed_time[i-1] + event_elapsed_time[i]
            new_case_elapsed_raw = prev_case_elapsed_raw + current_event_elapsed_raw
            
            # seconds_in_day[i] = (seconds_in_day[i-1] + event_elapsed_time[i]) % 86400
            new_seconds_in_day_raw = (prev_seconds_in_day_raw + current_event_elapsed_raw) % 86400
            
            # day_in_week[i] = (day_in_week[i-1] + round((seconds_in_day[i-1] + event_elapsed_time[i]) / 86400)) % 7
            days_to_add = round((prev_seconds_in_day_raw + current_event_elapsed_raw) / 86400)
            new_day_in_week_raw = (prev_day_in_week_raw + days_to_add) % 7
            
            # Transform back to encoded space
            new_case_elapsed_encoded = self.dataset.encoder_decoder.continuous_encoders["case_elapsed_time"].transform(
                [[new_case_elapsed_raw]]
            )[0][0]
            
            new_seconds_in_day_encoded = self.dataset.encoder_decoder.continuous_encoders["seconds_in_day"].transform(
                [[new_seconds_in_day_raw]]
            )[0][0]
            
            new_day_in_week_encoded = self.dataset.encoder_decoder.continuous_encoders["day_in_week"].transform(
                [[new_day_in_week_raw]]
            )[0][0]
            
            # Update the tensors using direct assignment with scalar values
            # Use with torch.no_grad() to ensure no gradient tracking
            with torch.no_grad():
                num_tensors[case_elapsed_idx][0, event_idx] = float(new_case_elapsed_encoded)
                num_tensors[seconds_in_day_idx][0, event_idx] = float(new_seconds_in_day_encoded)
                num_tensors[day_in_week_idx][0, event_idx] = float(new_day_in_week_encoded)
    
    def _detach_case(self, case):
        """
        Detach all tensors in a case tuple to avoid gradient issues in case_to_readable.
        Moves to CPU and clones to ensure tensors are completely independent.
            
        Returns:
            Detached case tuple
        """
        detached_cats = [t.detach().cpu().clone() for t in case[0]]
        detached_nums = [t.detach().cpu().clone() for t in case[1]]
        return (detached_cats, detached_nums)

