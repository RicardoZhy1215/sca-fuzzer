"""
SpecRL with hierarchical action space + action masking.
Uses inst_space for action_to_tuple and legality masks.
"""
import torch
import torch.nn as nn
from gymnasium import spaces
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from inst_space import (
    build_action_to_tuple as build_hi_action_to_tuple,
    compute_flat_action_mask,
    get_num_opcodes,
    get_num_regs,
    get_num_imms,
)


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

       from rvzr.isa_spec import InstructionSet
       instruction_set = InstructionSet("/home/hz25d/sca-fuzzer/base.json")
       opcode_size = len(set(spec.name.lower() for spec in instruction_set.instructions))
       reg_size = len(instruction_set.get_reg64_spec())

       # Instruction: (seq_size, 4) -> embed each of [opname, reg_src, reg_dst, imm]
       instr_box = obs_space.spaces.get("instruction")
       if instr_box is not None:
           if instr_vocab_sizes is not None:
               self.instr_vocab_sizes = [max(2, s) for s in instr_vocab_sizes]
           else:
               # vocab sizes: opcode ~1k+, reg ~20; use +2 for -1 padding
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
       batch = obs_dict["instruction"].shape[0]
       instr = obs_dict["instruction"]
       if self.instr_embeds is not None and self.instr_proj is not None:
           embs = []
           for i in range(4):
               e = self._safe_embed(instr[..., i], self.instr_embeds[i], i)
               embs.append(e)
           instr_feat = torch.cat(embs, dim=-1)
           instr_feat = self.instr_proj(instr_feat)
       else:
           instr_feat = torch.zeros(batch, self.seq_size, self.hidden_dim, device=instr.device)

       def _trace_norm(t: torch.Tensor) -> torch.Tensor:
           t = t.float().clamp(-1e9, 1e9)
           t = torch.where(t < 0, torch.zeros_like(t), t)
           return t

       h = _trace_norm(obs_dict["htrace"])
       c = _trace_norm(obs_dict["ctrace"])
       r = obs_dict["recovery_cycles"].float().clamp(-1, 1e6)
       u = obs_dict["transient_uops"].float().clamp(-1, 1e6)
       h = torch.where(h < 0, torch.zeros_like(h), h)
       c = torch.where(c < 0, torch.zeros_like(c), c)
       r = torch.where(r < 0, torch.zeros_like(r), r)
       u = torch.where(u < 0, torch.zeros_like(u), u)

       h_f = self.htrace_proj(h)
       c_f = self.ctrace_proj(c)
       r_f = self.recovery_proj(r)
       u_f = self.transient_proj(u)

       row = torch.cat([instr_feat, h_f, c_f, r_f, u_f], dim=-1)
       row = self.row_fusion(row)
       return row.mean(dim=1)


def _normalize_reg(reg_name: str) -> str:
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




def build_action_to_tuple() -> list:
   """
   Build action_to_tuple from inst_space (opcode, reg_src, reg_dst, imm).
   All actions are legal per OPCODE_OPERAND_SPEC; last action is end_game.
   """
   return build_hi_action_to_tuple()




class AutoregressiveInstructionHead(nn.Module):
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


class SpecRLHiModel(TorchModelV2, nn.Module):
   def __init__(self, obs_space, action_space, num_outputs, model_config, name):
       TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
       nn.Module.__init__(self)

       cc = model_config.get("custom_model_config", {})
       seq_size = cc.get("seq_size", 100)
       num_inputs = cc.get("num_inputs", 20)
       hidden_dim = cc.get("hidden_dim", 256)
       use_dict_obs = cc.get("use_dict_obs", True)

       self._seq_size = seq_size
       self._num_inputs = num_inputs
       original = getattr(obs_space, "original_space", obs_space)
       self.use_dict_obs = use_dict_obs and isinstance(original, spaces.Dict)

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
               original,
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
           action_to_tuple = build_action_to_tuple()
       self._action_to_tuple = action_to_tuple
       num_opcodes = get_num_opcodes()
       num_regs = get_num_regs()
       num_imms = get_num_imms()
       self._flat_action_mask = torch.tensor(
           compute_flat_action_mask(action_to_tuple), dtype=torch.bool
       )
       self.instruction_head = AutoregressiveInstructionHead(
           enc_dim,
           num_outputs,
           action_to_tuple=action_to_tuple,
           num_opcodes=num_opcodes - 1,
           num_regs=num_regs - 1,
           num_imms=num_imms,
           embed_dim=cc.get("ar_embed_dim", 64),
           head_hidden=cc.get("head_hidden", 128),
       )
       self.value_head = nn.Linear(enc_dim, 1)


   def _get_obs_tensors(self, input_dict: dict) -> tuple:
       obs = input_dict.get("obs_flat", input_dict.get("obs"))
       if isinstance(obs, dict) and "instruction" in obs:
           return {k: v.float() if v.dtype != torch.int64 else v for k, v in obs.items()}, True
       if isinstance(obs, torch.Tensor) and obs.dim() == 2:
           seq_size = getattr(self, "_seq_size", 100)
           num_inputs = getattr(self, "_num_inputs", 20)
           return _unflatten_obs(obs, seq_size, num_inputs), True


   @override(TorchModelV2)
   def forward(self, input_dict, state, seq_lens):
       obs, is_dict = self._get_obs_tensors(input_dict)
       if is_dict and self.encoder is not None:
           required = ["instruction", "htrace", "ctrace", "recovery_cycles", "transient_uops"]
           if all(k in obs for k in required):
               features = self.encoder(obs)
           else:
               flat = torch.cat([v.float().flatten(1) for v in obs.values()], dim=1)
               features = self.flat_backbone(flat)
       else:
           if isinstance(obs, dict):
               obs = torch.cat([v.float().flatten(1) for v in obs.values()], dim=1)
           features = self.flat_backbone(obs)

       action_mask = input_dict.get("action_mask")
       if action_mask is None:
           device = features.device
           mask = self._flat_action_mask.to(device)
           action_mask = mask.unsqueeze(0).expand(features.shape[0], -1)
       elif action_mask.dim() == 1:
           action_mask = action_mask.unsqueeze(0)
       if action_mask.shape[1] == self._flat_action_mask.shape[0]:
           static = self._flat_action_mask.to(action_mask.device).unsqueeze(0).expand_as(action_mask)
           action_mask = action_mask & static

       self._features = features
       logits = self.instruction_head(features, action_mask)
       return logits, state


   @override(TorchModelV2)
   def value_function(self):
       assert self._features is not None
       return self.value_head(self._features).squeeze(-1)


def register_specrl_hi_model():
   ModelCatalog.register_custom_model("SpecRLHiModel", SpecRLHiModel)
