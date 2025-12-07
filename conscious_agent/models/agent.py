"""
Main Conscious Agent Implementation
Integrates all novel components with frozen pretrained model
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional, Tuple

from .cognitive_attention import CognitiveMultiHeadAttention
from .self_model import HierarchicalSelfModel
from .curiosity import EpistemicCuriosityModule
from .value_system import MultiComponentValueSystem
from .generation_adapter import GenerationAdapter


class ConsciousAgent(nn.Module):
    """
    Conscious AI Agent with:
    - Frozen pretrained language model
    - Novel cognitive architecture (trainable)
    - Multi-component value system
    - Self-modeling capabilities
    - Epistemic curiosity
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.device = config['model']['device']
        
        # === FROZEN PRETRAINED MODEL ===
        print("Loading pretrained model...")
        self.pretrained_lm = AutoModelForCausalLM.from_pretrained(
            config['model']['base_model'],
            torch_dtype=getattr(torch, config['model']['dtype']),
            device_map="auto"
        )
        
        # Freeze pretrained model
        for param in self.pretrained_lm.parameters():
            param.requires_grad = False
        
        print(f"Pretrained model frozen: {config['model']['base_model']}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model']['base_model']
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get model dimensions
        self.hidden_dim = config['model']['hidden_dim']
        
        # === NOVEL TRAINABLE COMPONENTS ===
        
        # 1. Cognitive Multi-Head Attention
        self.cognitive_attention = CognitiveMultiHeadAttention(
            dim=self.hidden_dim,
            config=config['architecture']['cognitive_attention']
        )
        
        # 2. Hierarchical Self-Model
        self.self_model = HierarchicalSelfModel(
            dim=self.hidden_dim,
            config=config['architecture']['self_model']
        )
        
        # 3. Epistemic Curiosity System
        self.curiosity_module = EpistemicCuriosityModule(
            dim=self.hidden_dim,
            config=config['architecture']['curiosity']
        )
        
        # 4. Multi-Component Value System (with immutable core)
        self.value_system = MultiComponentValueSystem(
            dim=self.hidden_dim,
            config=config['architecture']['value_system']
        )
        
        # 5. Integration Layer
        self.integration_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 5, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
        # 6. Generation Adapter
        self.generation_adapter = GenerationAdapter(
            dim=self.hidden_dim
        )
        
        # 7. Thought Projector (Soft Prompting)
        # Projects cognitive state to LLM embedding space
        self.thought_projector = nn.Linear(
            self.hidden_dim, 
            self.pretrained_lm.config.hidden_size
        )
        
        # Internal state (persists across interactions)
        self.reset_internal_state()
        
        print(f"Conscious Agent initialized")
        self.print_parameter_count()
    
    def reset_internal_state(self):
        """Reset internal state for new episode"""
        self.internal_state = {
            'self_model_state': None,
            'curiosity_state': None,
            'interaction_history': [],
            'step': 0
        }
    
    def forward(
        self,
        text_input: str,
        human_state: Optional[Dict] = None,
        return_details: bool = False
    ) -> Dict:
        """
        Forward pass through conscious agent
        
        Args:
            text_input: Input text from human/environment
            human_state: Optional dict with human mental state
            return_details: Whether to return detailed internal states
        
        Returns:
            Dict with generated response and internal states
        """
        
        # === STEP 1: PRETRAINED MODEL PROCESSING (FROZEN) ===
        
        # Tokenize input
        inputs = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get hidden states from frozen pretrained model
        with torch.no_grad():
            pretrained_outputs = self.pretrained_lm(
                **inputs,
                output_hidden_states=True,
                use_cache=False
            )
            
            # Extract last layer hidden states
            hidden_states = pretrained_outputs.hidden_states[-1]  # [batch, seq, hidden]
        
        # === STEP 2: NOVEL COGNITIVE PROCESSING (TRAINABLE) ===
        
        # Prepare context for cognitive components
        context = self._prepare_context(hidden_states, human_state)
        
        # A. Cognitive Attention (different modes of thinking)
        cognitive_output = self.cognitive_attention(
            hidden_states,
            context=context
        )
        
        # B. Self-Model Update
        experience = {
            'perception': hidden_states.mean(dim=1),  # [batch, hidden]
            'context': cognitive_output.integrated
        }
        self_model_output = self.self_model(
            experience=experience,
            current_self=self.internal_state['self_model_state']
        )
        self.internal_state['self_model_state'] = self_model_output
        
        # C. Curiosity Computation
        curiosity_output = self.curiosity_module(
            state=hidden_states.mean(dim=1),
            self_knowledge=self_model_output,
            history=self.internal_state['curiosity_state']
        )
        self.internal_state['curiosity_state'] = curiosity_output
        
        # D. Value System Evaluation
        value_output = self.value_system.evaluate_state(
            state=hidden_states.mean(dim=1),
            human_state=human_state,
            self_model=self_model_output
        )
        
        # === STEP 3: INTEGRATION ===
        
        # Combine all cognitive components
        integrated_consciousness = self.integration_layer(
            torch.cat([
                cognitive_output.integrated,
                self_model_output.representation,
                curiosity_output.representation,
                value_output.representation,
                hidden_states.mean(dim=1)
            ], dim=-1)
        )
        
        # === STEP 4: GENERATION GUIDANCE ===
        
        generation_guidance = self.generation_adapter(
            integrated_consciousness=integrated_consciousness,
            value_constraints=value_output,
            curiosity_bias=curiosity_output
        )
        
        # === STEP 5: GENERATE RESPONSE ===
        
        generated_ids = self._generate_conscious(
            inputs=inputs,
            guidance=generation_guidance,
            thought_vector=integrated_consciousness
        )
        
        # Decode response
        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        
        # Remove input text from response
        if text_input in generated_text:
            generated_text = generated_text.replace(text_input, "").strip()
        
        # === STEP 6: UPDATE INTERNAL STATE ===
        
        self.internal_state['interaction_history'].append({
            'input': text_input,
            'response': generated_text,
            'cognitive_state': cognitive_output,
            'self_model': self_model_output,
            'curiosity': curiosity_output,
            'values': value_output
        })
        self.internal_state['step'] += 1
        
        # === RETURN ===
        
        result = {
            'response': generated_text,
            'generated_ids': generated_ids,
            'thought_vector': integrated_consciousness  # Exposed for auxiliary training
        }
        
        if return_details:
            result.update({
                'cognitive_output': cognitive_output,
                'self_model_output': self_model_output,
                'curiosity_output': curiosity_output,
                'value_output': value_output,
                'attention_weights': cognitive_output.attention_weights,
                'integrated_consciousness': integrated_consciousness
            })
        
        return result
    
    def _prepare_context(self, hidden_states: torch.Tensor, human_state: Optional[Dict]) -> Dict:
        """Prepare context for cognitive components"""
        
        context = {
            'hidden_states': hidden_states,
            'self_model': self.internal_state.get('self_model_state'),
            'curiosity': self.internal_state.get('curiosity_state'),
            'human_state': human_state or {},
            'step': self.internal_state['step']
        }
        
        # Add uncertainty estimate (for epistemic attention)
        if self.internal_state['curiosity_state'] is not None:
            context['uncertainty'] = self.internal_state['curiosity_state'].uncertainty
        else:
            context['uncertainty'] = torch.ones(hidden_states.size(0), 1).to(self.device)
        
        return context
    
    def _generate_conscious(
        self,
        inputs: Dict,
        guidance: 'GenerationGuidance',
        thought_vector: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate text using pretrained model, guided by consciousness
        """
        
        # Use frozen pretrained model for generation
        with torch.no_grad():
            if thought_vector is not None:
                # === SOFT PROMPTING ===
                # 1. Project thought to embedding space
                # [batch, 1, embed_dim]
                thought_embedding = self.thought_projector(thought_vector).unsqueeze(1)
                
                # 2. Get original input embeddings
                input_ids = inputs['input_ids']
                inputs_embeds = self.pretrained_lm.get_input_embeddings()(input_ids)
                
                # 3. Concatenate thought + text
                combined_embeds = torch.cat([thought_embedding, inputs_embeds], dim=1)
                
                # 4. Adjust attention mask
                attention_mask = inputs['attention_mask']
                thought_mask = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
                combined_mask = torch.cat([thought_mask, attention_mask], dim=1)
                
                generated_ids = self.pretrained_lm.generate(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_mask,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=guidance.temperature.item(),
                    top_p=guidance.top_p.item(),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            else:
                generated_ids = self.pretrained_lm.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=guidance.temperature.item(),
                    top_p=guidance.top_p.item(),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
        
        return generated_ids
    
    def forward_train(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        human_state: Optional[Dict] = None
    ) -> Dict:
        """
        Forward pass for Supervised Pre-training (No Generation)
        Returns logits and internal states for loss computation
        """
        
        # === STEP 1: GET HIDDEN STATES (FROZEN) ===
        with torch.no_grad():
            pretrained_outputs = self.pretrained_lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False
            )
            hidden_states = pretrained_outputs.hidden_states[-1]
            
        # === STEP 2: COGNITIVE PROCESSING ===
        context = self._prepare_context(hidden_states, human_state)
        
        # A. Cognitive Attention
        cognitive_output = self.cognitive_attention(hidden_states, context=context)
        
        # B. Self-Model
        experience = {
            'perception': hidden_states.mean(dim=1),
            'context': cognitive_output.integrated
        }
        self_model_output = self.self_model(
            experience=experience,
            current_self=self.internal_state['self_model_state']
        )
        
        # C. Curiosity
        curiosity_output = self.curiosity_module(
            state=hidden_states.mean(dim=1),
            self_knowledge=self_model_output,
            history=self.internal_state['curiosity_state']
        )
        
        # D. Values
        value_output = self.value_system.evaluate_state(
            state=hidden_states.mean(dim=1),
            human_state=human_state,
            self_model=self_model_output
        )
        
        # === STEP 3: INTEGRATION ===
        integrated_consciousness = self.integration_layer(
            torch.cat([
                cognitive_output.integrated,
                self_model_output.representation,
                curiosity_output.representation,
                value_output.representation,
                hidden_states.mean(dim=1)
            ], dim=-1)
        )
        
        # === STEP 4: SOFT PROMPTING INJECTION ===
        # Project thought to embedding space
        thought_embedding = self.thought_projector(integrated_consciousness).unsqueeze(1)
        
        # Get original input embeddings
        inputs_embeds = self.pretrained_lm.get_input_embeddings()(input_ids)
        
        # Concatenate thought + text
        combined_embeds = torch.cat([thought_embedding, inputs_embeds], dim=1)
        
        # Adjust attention mask
        thought_mask = torch.ones((attention_mask.shape[0], 1), device=self.device)
        combined_mask = torch.cat([thought_mask, attention_mask], dim=1)
        
        # === STEP 5: FORWARD PASS (NO GENERATION) ===
        outputs = self.pretrained_lm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask
        )
        
        return {
            'logits': outputs.logits,
            'thought_vector': integrated_consciousness,
            'curiosity_output': curiosity_output,
            'value_output': value_output,
            'self_model_output': self_model_output
        }

    def print_parameter_count(self):
        """Print parameter counts"""
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.pretrained_lm.parameters())
        total = trainable + frozen
        
        print(f"\n{'='*60}")
        print(f"Parameter Count:")
        print(f"  Trainable (novel components): {trainable:,} ({trainable/1e6:.1f}M)")
        print(f"  Frozen (pretrained model): {frozen:,} ({frozen/1e9:.2f}B)")
        print(f"  Total: {total:,} ({total/1e9:.2f}B)")
        print(f"  Percentage trainable: {100 * trainable / total:.2f}%")
        print(f"{'='*60}\n")
    
    def get_trainable_parameters(self):
        """Get only trainable parameters (for optimizer)"""
        return [p for p in self.parameters() if p.requires_grad]