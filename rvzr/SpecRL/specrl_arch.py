"""
SpecRL architecture: ObsEncoder + AutoregressiveInstructionHead for SpecEnv.


- ObsEncoder: multi-modal encoder for Dict obs (instruction, htrace, ctrace, recovery_cycles, transient_uops)
 Supports large instruction vocab via embeddings; mean pooling over sequence.
- AutoregressiveInstructionHead: opcode -> reg_src -> reg_dst -> imm (mini-AlphaStar style).


Usage:
 from rvzr.SpecRL.specrl_arch import register_specrl_model, build_action_to_tuple
 register_specrl_model()
 action_to_tuple = build_action_to_tuple(instruction_space, opcode_vocab, reg_vocab)
 config.training(model={"custom_model": "SpecRLModel", "custom_model_config": {
     "seq_size": 100, "num_inputs": 20,
     "action_to_tuple": action_to_tuple,
     "hidden_dim": 256,
 }})
"""


import torch
import torch.nn as nn
from gymnasium import spaces
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from sanity_check_SpecEnv import SpecEnv


def _get_flat_input_size(obs_space, seq_size=100, num_inputs=20):
   """Compute flattened obs size (same as model.py)."""
   if isinstance(obs_space, spaces.Dict):
       from ray.rllib.utils.spaces.space_utils import flatten_space
       flat_space = flatten_space(obs_space)
       return flat_space.shape[0]
   return obs_space.shape[0]




def _unflatten_obs(flat_obs: torch.Tensor, seq_size: int, num_inputs: int) -> dict:
   """
   Reconstruct dict obs from RLlib's flattened observation.
   Order: instruction(seq,4), htrace(seq,K), ctrace(seq,K), recovery(seq,K), transient(seq,K)
   """
   b = flat_obs.shape[0]
   i = 0
   instr = flat_obs[:, i : i + seq_size * 4].view(b, seq_size, 4).long()
   i += seq_size * 4
   htrace = flat_obs[:, i : i + seq_size * num_inputs].view(b, seq_size, num_inputs)
   i += seq_size * num_inputs
   ctrace = flat_obs[:, i : i + seq_size * num_inputs].view(b, seq_size, num_inputs)
   i += seq_size * num_inputs
   recovery = flat_obs[:, i : i + seq_size * num_inputs].view(b, seq_size, num_inputs)
   i += seq_size * num_inputs
   transient = flat_obs[:, i : i + seq_size * num_inputs].view(b, seq_size, num_inputs)
   return {
       "instruction": instr,
       "htrace": htrace,
       "ctrace": ctrace,
       "recovery_cycles": recovery,
       "transient_uops": transient,
   }




# ---------------------------------------------------------------------------
# ObsEncoder: encodes Dict observation (instruction, htrace, ctrace, ...)
# ---------------------------------------------------------------------------


class ObsEncoder(nn.Module):
   """
   Multi-modal encoder for SpecEnv observation.
   Handles: instruction (seq,4), htrace (seq,K), ctrace (seq,K), recovery_cycles, transient_uops.
   """


   def __init__(
       self,
       obs_space: spaces.Dict,
       seq_size: int = 100,
       num_inputs: int = 20,
       hidden_dim: int = 256,
       instruction_embed_dim: int = 64,
       trace_embed_dim: int = 32,
       instr_vocab_sizes: list = None,
   ):
       super().__init__()
       self.seq_size = seq_size
       self.num_inputs = num_inputs
       self.hidden_dim = hidden_dim

        

       # Instruction: (seq_size, 4) -> embed each of [opname, reg_src, reg_dst, imm]
       instr_box = obs_space.spaces.get("instruction")
       if instr_box is not None:
           if instr_vocab_sizes is not None:
               self.instr_vocab_sizes = [max(2, s) for s in instr_vocab_sizes]
           else:
               # vocab sizes: opcode ~1k+, reg ~20; use +2 for -1 padding
               opcode_size = SpecEnv._get_opcode_size()
               reg_size = SpecEnv._get_reg_size()
               self.instr_vocab_sizes = [
                   opcode_size + 2,
                   reg_size + 2,
                   reg_size + 2,
                   4,  # imm
               ]
           self.instr_embeds = nn.ModuleList([
               nn.Embedding(max(2, s), instruction_embed_dim, padding_idx=0)
               for s in self.instr_vocab_sizes
           ])
           self.instr_proj = nn.Linear(4 * instruction_embed_dim, hidden_dim)
       else:
           self.instr_embeds = None
           self.instr_proj = None


       # Trace-like (htrace, ctrace): (seq_size, num_inputs), int64
       # Use simple linear projection after normalization (addresses can be large)
       trace_in = num_inputs
       self.htrace_proj = nn.Sequential(
           nn.Linear(trace_in, trace_embed_dim),
           nn.LayerNorm(trace_embed_dim),
           nn.ReLU(),
       )
       self.ctrace_proj = nn.Sequential(
           nn.Linear(trace_in, trace_embed_dim),
           nn.LayerNorm(trace_embed_dim),
           nn.ReLU(),
       )
       self.recovery_proj = nn.Sequential(
           nn.Linear(trace_in, trace_embed_dim),
           nn.LayerNorm(trace_embed_dim),
           nn.ReLU(),
       )
       self.transient_proj = nn.Sequential(
           nn.Linear(trace_in, trace_embed_dim),
           nn.LayerNorm(trace_embed_dim),
           nn.ReLU(),
       )


       # Per-row: fuse instr + 4 trace features -> row_dim; mean pool over sequence
       row_dim = (hidden_dim if self.instr_proj is not None else 0) + 4 * trace_embed_dim
       self.row_fusion = nn.Sequential(
           nn.Linear(row_dim, hidden_dim),
           nn.LayerNorm(hidden_dim),
           nn.ReLU(),
       )
       self.encoder_out_dim = hidden_dim


   def _safe_embed(self, x: torch.Tensor, embed: nn.Embedding, vocab_idx: int) -> torch.Tensor:
       """Embed with clamp for out-of-range; -1 -> 0 (padding)."""
       x = x.long().clamp(-1, embed.num_embeddings - 1)
       x = (x + 1).clamp(0, embed.num_embeddings - 1)
       return embed(x)


   def forward(self, obs_dict: dict) -> torch.Tensor:
       """
       obs_dict: dict with keys instruction, htrace, ctrace, recovery_cycles, transient_uops.
       Each value: (B, seq_size, ...).
       Returns: (B, encoder_out_dim).
       """
       batch = obs_dict["instruction"].shape[0]


       # Instruction encoding
       instr = obs_dict["instruction"]  # (B, seq, 4)
       if self.instr_embeds is not None and self.instr_proj is not None:
           embs = []
           for i in range(4):
               e = self._safe_embed(instr[..., i], self.instr_embeds[i], i)
               embs.append(e)
           instr_feat = torch.cat(embs, dim=-1)  # (B, seq, 4*embed_dim)
           instr_feat = self.instr_proj(instr_feat)  # (B, seq, hidden_dim)
       else:
           instr_feat = torch.zeros(batch, self.seq_size, self.hidden_dim, device=instr.device)


       # Trace encoding; handle -1 / large values via clamp+normalize
       def _trace_norm(t: torch.Tensor) -> torch.Tensor:
           t = t.float().clamp(-1e9, 1e9)
           t = torch.where(t < 0, torch.zeros_like(t), t)
           return t


       h = _trace_norm(obs_dict["htrace"])
       c = _trace_norm(obs_dict["ctrace"])
       r = obs_dict["recovery_cycles"].float().clamp(-1, 1e6)
       u = obs_dict["transient_uops"].float().clamp(-1, 1e6)
       # -1 -> 0 for proj
       h = torch.where(h < 0, torch.zeros_like(h), h)
       c = torch.where(c < 0, torch.zeros_like(c), c)
       r = torch.where(r < 0, torch.zeros_like(r), r)
       u = torch.where(u < 0, torch.zeros_like(u), u)


       h_f = self.htrace_proj(h)
       c_f = self.ctrace_proj(c)
       r_f = self.recovery_proj(r)
       u_f = self.transient_proj(u)


       row = torch.cat([instr_feat, h_f, c_f, r_f, u_f], dim=-1)
       row = self.row_fusion(row)  # (B, seq, hidden_dim)
       return row.mean(dim=1)




# ---------------------------------------------------------------------------
# AutoregressiveInstructionHead: mini-AlphaStar style, opcode -> reg_src -> reg_dst -> imm
# ---------------------------------------------------------------------------


def _normalize_reg(reg_name: str) -> str:
   """Normalize register name to 64-bit form."""
   m = {
       "eax": "rax", "ax": "rax", "al": "rax", "ah": "rax",
       "ebx": "rbx", "bx": "rbx", "bl": "rbx", "bh": "rbx",
       "ecx": "rcx", "cx": "rcx", "cl": "rcx", "ch": "rcx",
       "edx": "rdx", "dx": "rdx", "dl": "rdx", "dh": "rdx",
       "esi": "rsi", "si": "rsi", "sil": "rsi",
       "edi": "rdi", "di": "rdi", "dil": "rdi",
       "rip": "rip",
   }
   return m.get(reg_name, reg_name)




def build_action_to_tuple(instruction_space, opcode_vocab: list, reg_vocab: list) -> list:
   """
   Build mapping: action_index -> (opcode_id, reg_src_id, reg_dst_id, imm_id).
   opcode_id: 0 = end_game, 1..K = opcode_vocab indices (shifted).
   reg_src_id, reg_dst_id: 0 = N/A, 1..R = reg_vocab indices.
   imm_id: 0 = no imm, 1 = has imm.
   """
   from rvzr.tc_components.instruction import MemoryOp


   def _reg_to_id(op, reg_vocab) -> int:
       val = op.value.lower() if hasattr(op, "value") else str(op).lower()
       val_norm = _normalize_reg(val)
       return reg_vocab.index(val_norm) + 1 if val_norm in reg_vocab else 0


   result = []
   for inst in instruction_space:
       name_lower = inst.name.lower()
       opcode_id = opcode_vocab.index(name_lower) + 1 if name_lower in opcode_vocab else 0
       reg_src_id = 0
       reg_dst_id = 0
       for op in inst.get_reg_operands(include_implicit=True):
           idx = _reg_to_id(op, reg_vocab)
           if idx and op.src and reg_src_id == 0:
               reg_src_id = idx
           if idx and op.dest and reg_dst_id == 0:
               reg_dst_id = idx
       for op in inst.operands:
           if isinstance(op, MemoryOp) and op.get_base_register() is not None:
               base = op.get_base_register()
               idx = _reg_to_id(base, reg_vocab)
               if idx and reg_src_id == 0:
                   reg_src_id = idx
               elif idx and reg_dst_id == 0:
                   reg_dst_id = idx
       imm_id = 1 if inst.get_imm_operands() else 0
       result.append((opcode_id, reg_src_id, reg_dst_id, imm_id))
   result.append((0, 0, 0, 0))  # end_game: opcode 0
   return result




class AutoregressiveInstructionHead(nn.Module):
   """
   Autoregressive action head (mini-AlphaStar style).
   Samples: opcode -> reg_src -> reg_dst -> imm. Each head conditions on previous.
   Outputs logits for full Discrete action space by composing per-head log probs.
   """


   def __init__(
       self,
       input_dim: int,
       num_actions: int,
       action_to_tuple: list,
       num_opcodes: int,
       num_regs: int,
       num_imms: int = 2,
       embed_dim: int = 64,
       head_hidden: int = 128,
   ):
       super().__init__()
       self.num_actions = num_actions
       self.action_to_tuple = action_to_tuple


       # opcode: 0=end, 1..num_opcodes
       self.num_opcodes = num_opcodes + 1
       self.num_regs = num_regs + 1
       self.num_imms = max(2, num_imms)


       self.opcode_embed = nn.Embedding(self.num_opcodes, embed_dim, padding_idx=0)
       self.reg_embed = nn.Embedding(self.num_regs, embed_dim, padding_idx=0)
       self.imm_embed = nn.Embedding(self.num_imms, embed_dim, padding_idx=0)


       self.opcode_head = nn.Sequential(
           nn.Linear(input_dim, head_hidden),
           nn.ReLU(),
           nn.Linear(head_hidden, self.num_opcodes),
       )
       self.reg_src_head = nn.Sequential(
           nn.Linear(input_dim + embed_dim, head_hidden),
           nn.ReLU(),
           nn.Linear(head_hidden, self.num_regs),
       )
       self.reg_dst_head = nn.Sequential(
           nn.Linear(input_dim + embed_dim * 2, head_hidden),
           nn.ReLU(),
           nn.Linear(head_hidden, self.num_regs),
       )
       self.imm_head = nn.Sequential(
           nn.Linear(input_dim + embed_dim * 3, head_hidden),
           nn.ReLU(),
           nn.Linear(head_hidden, self.num_imms),
       )


   def _get_reg_src_input(self, features: torch.Tensor, opcode_id: torch.Tensor) -> torch.Tensor:
       op_emb = self.opcode_embed(opcode_id.clamp(0, self.num_opcodes - 1))
       return torch.cat([features, op_emb], dim=-1)


   def _get_reg_dst_input(
       self, features: torch.Tensor, opcode_id: torch.Tensor, reg_src_id: torch.Tensor
   ) -> torch.Tensor:
       op_emb = self.opcode_embed(opcode_id.clamp(0, self.num_opcodes - 1))
       rs_emb = self.reg_embed(reg_src_id.clamp(0, self.num_regs - 1))
       return torch.cat([features, op_emb, rs_emb], dim=-1)


   def _get_imm_input(
       self,
       features: torch.Tensor,
       opcode_id: torch.Tensor,
       reg_src_id: torch.Tensor,
       reg_dst_id: torch.Tensor,
   ) -> torch.Tensor:
       op_emb = self.opcode_embed(opcode_id.clamp(0, self.num_opcodes - 1))
       rs_emb = self.reg_embed(reg_src_id.clamp(0, self.num_regs - 1))
       rd_emb = self.reg_embed(reg_dst_id.clamp(0, self.num_regs - 1))
       return torch.cat([features, op_emb, rs_emb, rd_emb], dim=-1)


   def forward_logits(self, features: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
       """
       Compute logits for all actions via autoregressive chain rule.
       log p(a) = log p(opcode) + log p(reg_src|opcode) + log p(reg_dst|...) + log p(imm|...)
       """
       B = features.shape[0]
       device = features.device


       opcode_logits = self.opcode_head(features)
       opcode_logp = torch.log_softmax(opcode_logits, dim=-1)


       logits_out = torch.empty(B, self.num_actions, device=device, dtype=features.dtype)


       for a in range(self.num_actions):
           o, rs, rd, imm = self.action_to_tuple[a]
           o = min(o, self.num_opcodes - 1)
           rs = min(rs, self.num_regs - 1)
           rd = min(rd, self.num_regs - 1)
           imm = min(imm, self.num_imms - 1)


           logp = opcode_logp[:, o].clone()


           reg_src_in = self._get_reg_src_input(features, torch.full((B,), o, device=device, dtype=torch.long))
           reg_src_logits = self.reg_src_head(reg_src_in)
           logp = logp + torch.log_softmax(reg_src_logits, dim=-1)[:, rs]


           reg_dst_in = self._get_reg_dst_input(
               features,
               torch.full((B,), o, device=device, dtype=torch.long),
               torch.full((B,), rs, device=device, dtype=torch.long),
           )
           reg_dst_logits = self.reg_dst_head(reg_dst_in)
           logp = logp + torch.log_softmax(reg_dst_logits, dim=-1)[:, rd]


           imm_in = self._get_imm_input(
               features,
               torch.full((B,), o, device=device, dtype=torch.long),
               torch.full((B,), rs, device=device, dtype=torch.long),
               torch.full((B,), rd, device=device, dtype=torch.long),
           )
           imm_logits = self.imm_head(imm_in)
           logp = logp + torch.log_softmax(imm_logits, dim=-1)[:, imm]


           logits_out[:, a] = logp


       if action_mask is not None:
           logits_out = logits_out.masked_fill(~action_mask.bool(), float("-inf"))
       return logits_out


   def forward(self, features: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
       return self.forward_logits(features, action_mask)




# ---------------------------------------------------------------------------
# SpecRLModel: full model with ObsEncoder + AutoregressiveInstructionHead
# ---------------------------------------------------------------------------


class SpecRLModel(TorchModelV2, nn.Module):
   """
   SpecRL policy model: ObsEncoder -> AutoregressiveInstructionHead (+ value head).
   Works with Dict obs when available; falls back to flat MLP otherwise.
   """


   def __init__(self, obs_space, action_space, num_outputs, model_config, name):
       TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
       nn.Module.__init__(self)

       
       cc = model_config.get("custom_model_config", {})
       # Derive from obs_space so unflatten matches RLlib's flatten order
       seq_size = cc.get("seq_size", 100)
       num_inputs = cc.get("num_inputs", 20)
       hidden_dim = cc.get("hidden_dim", 256)
       use_dict_obs = cc.get("use_dict_obs", True)


       self._seq_size = seq_size
       self._num_inputs = num_inputs
       self.use_dict_obs = use_dict_obs and isinstance(obs_space, spaces.Dict)


       flat_size = _get_flat_input_size(obs_space)
       self.flat_backbone = nn.Sequential(
           nn.Linear(flat_size, hidden_dim),
           nn.LayerNorm(hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, hidden_dim),
           nn.LayerNorm(hidden_dim),
           nn.ReLU(),
       )
       if self.use_dict_obs:
           self.encoder = ObsEncoder(
               obs_space,
               seq_size=seq_size,
               num_inputs=num_inputs,
               hidden_dim=hidden_dim,
               instruction_embed_dim=cc.get("instruction_embed_dim", 64),
               trace_embed_dim=cc.get("trace_embed_dim", 32),
               instr_vocab_sizes=cc.get("instr_vocab_sizes"),
           )
           enc_dim = self.encoder.encoder_out_dim
       else:
           self.encoder = None
           enc_dim = hidden_dim


       action_to_tuple = cc.get("action_to_tuple")
       if action_to_tuple is None:
           raise ValueError("SpecRLModel requires custom_model_config['action_to_tuple']")
       num_opcodes = cc.get("num_opcodes") or max((t[0] for t in action_to_tuple), default=1)
       num_regs = cc.get("num_regs") or max(
           max(t[1] for t in action_to_tuple), max(t[2] for t in action_to_tuple), 1
       )
       self.instruction_head = AutoregressiveInstructionHead(
           enc_dim,
           num_outputs,
           action_to_tuple=action_to_tuple,
           num_opcodes=num_opcodes,
           num_regs=num_regs,
           num_imms=cc.get("num_imms", 2),
           embed_dim=cc.get("ar_embed_dim", 64),
           head_hidden=cc.get("head_hidden", 128),
       )
       self.value_head = nn.Linear(enc_dim, 1)


   def _get_obs_tensors(self, input_dict: dict) -> tuple:
       """
       Return (obs_dict_or_flat, is_dict).
       When RLlib flattens Dict obs, we reconstruct the dict from obs_flat.
       """
       obs = input_dict.get("obs_flat", input_dict.get("obs"))
       if isinstance(obs, dict) and "instruction" in obs:
           return {k: v.float() if v.dtype != torch.int64 else v for k, v in obs.items()}, True
       # Reconstruct dict from flattened obs (RLlib DictFlatteningPreprocessor)
       if isinstance(obs, torch.Tensor) and obs.dim() == 2:
           seq_size = getattr(self, "_seq_size", 100)
           num_inputs = getattr(self, "_num_inputs", 20)
           return _unflatten_obs(obs, seq_size, num_inputs), True


   @override(TorchModelV2)
   def forward(self, input_dict, state, seq_lens):
       obs, is_dict = self._get_obs_tensors(input_dict)
       action_mask = input_dict.get("action_mask")


       if is_dict and self.encoder is not None:
           # Ensure all keys present; handle preprocessor flattening
           required = ["instruction", "htrace", "ctrace", "recovery_cycles", "transient_uops"]
           if all(k in obs for k in required):
               features = self.encoder(obs)
           else:
               # Flatten fallback
               flat = torch.cat([v.float().flatten(1) for v in obs.values()], dim=1)
               features = self.flat_backbone(flat)
       else:
           if isinstance(obs, dict):
               obs = torch.cat([v.float().flatten(1) for v in obs.values()], dim=1)
           features = self.flat_backbone(obs)


       self._features = features
       logits = self.instruction_head(features, action_mask)
       return logits, state


   @override(TorchModelV2)
   def value_function(self):
       assert self._features is not None
       return self.value_head(self._features).squeeze(-1)




def register_specrl_model():
   ModelCatalog.register_custom_model("SpecRLModel", SpecRLModel)



