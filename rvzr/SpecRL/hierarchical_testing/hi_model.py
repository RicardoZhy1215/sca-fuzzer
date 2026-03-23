"""
SpecRL with hierarchical action space: (opcode, reg_src, reg_dst, imm).

- ObsEncoder: observation → features
- HierarchicalActionHead: 4 heads, autoregressive sampling with inst_space masks
- SpecRLHierarchicalModel: RLlib model for PPO
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils.annotations import override

from inst_space import (
    get_opcode_mask,
    get_reg_src_mask,
    get_reg_dst_mask,
    get_imm_mask,
    get_num_opcodes,
    get_num_regs,
    get_num_imms,
)


def _get_flat_input_size(obs_space, seq_size=100, num_inputs=20):
    """Compute flattened obs size."""
    if isinstance(obs_space, spaces.Dict):
        from ray.rllib.utils.spaces.space_utils import flatten_space
        flat_space = flatten_space(obs_space)
        return flat_space.shape[0]
    return obs_space.shape[0]


def _unflatten_obs(flat_obs: torch.Tensor, seq_size: int, num_inputs: int) -> dict:
    """Reconstruct dict obs from RLlib's flattened observation."""
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

        instr_box = obs_space.spaces.get("instruction")
        if instr_box is not None:
            if instr_vocab_sizes is not None:
                self.instr_vocab_sizes = [max(2, s) for s in instr_vocab_sizes]
            else:
                self.instr_vocab_sizes = [
                    opcode_size + 2,
                    reg_size + 2,
                    reg_size + 2,
                    4,
                ]
            self.instr_embeds = nn.ModuleList([
                nn.Embedding(max(2, s), instruction_embed_dim, padding_idx=0)
                for s in self.instr_vocab_sizes
            ])
            self.instr_proj = nn.Linear(4 * instruction_embed_dim, hidden_dim)
        else:
            self.instr_embeds = None
            self.instr_proj = None

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

        row_dim = (hidden_dim if self.instr_proj is not None else 0) + 4 * trace_embed_dim
        self.row_fusion = nn.Sequential(
            nn.Linear(row_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.encoder_out_dim = hidden_dim

    def _safe_embed(self, x: torch.Tensor, embed: nn.Embedding, vocab_idx: int) -> torch.Tensor:
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


# ---------------------------------------------------------------------------
# Hierarchical action space: 4 heads
# ---------------------------------------------------------------------------

def get_hierarchical_action_space() -> spaces.Tuple:
    """Returns gymnasium Tuple action space: (opcode, reg_src, reg_dst, imm)."""
    n_op = get_num_opcodes()
    n_reg = get_num_regs()
    n_imm = get_num_imms()
    return spaces.Tuple((
        spaces.Discrete(n_op),
        spaces.Discrete(n_reg),
        spaces.Discrete(n_reg),
        spaces.Discrete(n_imm),
    ))


class HierarchicalActionHead(nn.Module):
    """
    Four heads: opcode -> reg_src -> reg_dst -> imm. Autoregressive sampling
    with per-step legality masks from inst_space.
    """

    def __init__(
        self,
        input_dim: int,
        num_opcodes: int,
        num_regs: int,
        num_imms: int = 2,
        embed_dim: int = 64,
        head_hidden: int = 128,
    ):
        super().__init__()
        self.opcode_space_size = num_opcodes + 1
        self.reg_space_size = num_regs + 1
        self.imm_space_size = max(2, num_imms)

        self.opcode_embed = nn.Embedding(self.opcode_space_size, embed_dim, padding_idx=0)
        self.reg_embed = nn.Embedding(self.reg_space_size, embed_dim, padding_idx=0)
        self.imm_embed = nn.Embedding(self.imm_space_size, embed_dim, padding_idx=0)

        self.opcode_head = nn.Sequential(
            nn.Linear(input_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, self.opcode_space_size),
        )
        self.reg_src_head = nn.Sequential(
            nn.Linear(input_dim + embed_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, self.reg_space_size),
        )
        self.reg_dst_head = nn.Sequential(
            nn.Linear(input_dim + embed_dim * 2, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, self.reg_space_size),
        )
        self.imm_head = nn.Sequential(
            nn.Linear(input_dim + embed_dim * 3, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, self.imm_space_size),
        )

    def _apply_mask(self, logits: torch.Tensor, mask: list, device: torch.device) -> torch.Tensor:
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
        if mask_t.dim() == 1:
            mask_t = mask_t.unsqueeze(0).expand(logits.shape[0], -1)
        return logits.masked_fill(~mask_t, float("-inf"))

    def forward_logits(
        self,
        features: torch.Tensor,
        opcode_id: torch.Tensor = None,
        reg_src_id: torch.Tensor = None,
        reg_dst_id: torch.Tensor = None,
    ) -> tuple:
        B = features.shape[0]
        device = features.device

        opcode_logits = self.opcode_head(features)
        opcode_logits = self._apply_mask(opcode_logits, get_opcode_mask(), device)

        if opcode_id is None:
            return (opcode_logits, None, None, None)

        op_emb = self.opcode_embed(opcode_id.clamp(0, self.opcode_space_size - 1))
        reg_src_in = torch.cat([features, op_emb], dim=-1)
        reg_src_logits = self.reg_src_head(reg_src_in)
        for b in range(B):
            reg_src_logits[b] = self._apply_mask(
                reg_src_logits[b : b + 1], get_reg_src_mask(int(opcode_id[b].item())), device
            ).squeeze(0)

        if reg_src_id is None:
            return (opcode_logits, reg_src_logits, None, None)

        rs_emb = self.reg_embed(reg_src_id.clamp(0, self.reg_space_size - 1))
        reg_dst_in = torch.cat([features, op_emb, rs_emb], dim=-1)
        reg_dst_logits = self.reg_dst_head(reg_dst_in)
        for b in range(B):
            reg_dst_logits[b] = self._apply_mask(
                reg_dst_logits[b : b + 1],
                get_reg_dst_mask(int(opcode_id[b].item()), int(reg_src_id[b].item())),
                device,
            ).squeeze(0)

        if reg_dst_id is None:
            return (opcode_logits, reg_src_logits, reg_dst_logits, None)

        rd_emb = self.reg_embed(reg_dst_id.clamp(0, self.reg_space_size - 1))
        imm_in = torch.cat([features, op_emb, rs_emb, rd_emb], dim=-1)
        imm_logits = self.imm_head(imm_in)
        for b in range(B):
            imm_logits[b] = self._apply_mask(
                imm_logits[b : b + 1],
                get_imm_mask(int(opcode_id[b].item()), int(reg_src_id[b].item())),
                device,
            ).squeeze(0)

        return (opcode_logits, reg_src_logits, reg_dst_logits, imm_logits)

    def forward_sample(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
        return_logits: bool = True,
    ) -> dict:
        B = features.shape[0]
        device = features.device

        log_probs = []
        entropies = []
        all_logits = {}

        opcode_logits = self.opcode_head(features)
        opcode_logits = self._apply_mask(opcode_logits, get_opcode_mask(), device)
        all_logits["opcode"] = opcode_logits
        opcode_logp = F.log_softmax(opcode_logits, dim=-1)
        opcode_probs = opcode_logp.exp()

        if deterministic:
            opcode_id = opcode_probs.argmax(dim=-1)
        else:
            opcode_id = torch.multinomial(opcode_probs + 1e-10, num_samples=1).squeeze(-1)
        log_probs.append(opcode_logp.gather(1, opcode_id.unsqueeze(-1)).squeeze(-1))
        entropies.append(-(opcode_probs * opcode_logp).sum(dim=-1))

        reg_src_id = torch.zeros(B, dtype=torch.long, device=device)
        reg_dst_id = torch.zeros(B, dtype=torch.long, device=device)
        imm_id = torch.zeros(B, dtype=torch.long, device=device)

        op_emb = self.opcode_embed(opcode_id.clamp(0, self.opcode_space_size - 1))
        reg_src_in = torch.cat([features, op_emb], dim=-1)

        reg_src_logits = self.reg_src_head(reg_src_in)
        for b in range(B):
            reg_src_logits[b] = self._apply_mask(
                reg_src_logits[b : b + 1], get_reg_src_mask(int(opcode_id[b].item())), device
            ).squeeze(0)
        all_logits["reg_src"] = reg_src_logits
        reg_src_logp = F.log_softmax(reg_src_logits, dim=-1)
        reg_src_probs = reg_src_logp.exp()
        if deterministic:
            reg_src_id = reg_src_probs.argmax(dim=-1)
        else:
            reg_src_id = torch.multinomial(reg_src_probs + 1e-10, num_samples=1).squeeze(-1)
        log_probs.append(reg_src_logp.gather(1, reg_src_id.unsqueeze(-1)).squeeze(-1))
        entropies.append(-(reg_src_probs * reg_src_logp).sum(dim=-1))

        rs_emb = self.reg_embed(reg_src_id.clamp(0, self.reg_space_size - 1))
        reg_dst_in = torch.cat([features, op_emb, rs_emb], dim=-1)

        reg_dst_logits = self.reg_dst_head(reg_dst_in)
        for b in range(B):
            reg_dst_logits[b] = self._apply_mask(
                reg_dst_logits[b : b + 1],
                get_reg_dst_mask(int(opcode_id[b].item()), int(reg_src_id[b].item())),
                device,
            ).squeeze(0)
        all_logits["reg_dst"] = reg_dst_logits
        reg_dst_logp = F.log_softmax(reg_dst_logits, dim=-1)
        reg_dst_probs = reg_dst_logp.exp()
        if deterministic:
            reg_dst_id = reg_dst_probs.argmax(dim=-1)
        else:
            reg_dst_id = torch.multinomial(reg_dst_probs + 1e-10, num_samples=1).squeeze(-1)
        log_probs.append(reg_dst_logp.gather(1, reg_dst_id.unsqueeze(-1)).squeeze(-1))
        entropies.append(-(reg_dst_probs * reg_dst_logp).sum(dim=-1))

        rd_emb = self.reg_embed(reg_dst_id.clamp(0, self.reg_space_size - 1))
        imm_in = torch.cat([features, op_emb, rs_emb, rd_emb], dim=-1)

        imm_logits = self.imm_head(imm_in)
        for b in range(B):
            imm_logits[b] = self._apply_mask(
                imm_logits[b : b + 1],
                get_imm_mask(int(opcode_id[b].item()), int(reg_src_id[b].item())),
                device,
            ).squeeze(0)
        all_logits["imm"] = imm_logits
        imm_logp = F.log_softmax(imm_logits, dim=-1)
        imm_probs = imm_logp.exp()
        if deterministic:
            imm_id = imm_probs.argmax(dim=-1)
        else:
            imm_id = torch.multinomial(imm_probs + 1e-10, num_samples=1).squeeze(-1)
        log_probs.append(imm_logp.gather(1, imm_id.unsqueeze(-1)).squeeze(-1))
        entropies.append(-(imm_probs * imm_logp).sum(dim=-1))

        total_log_prob = sum(log_probs)
        total_entropy = sum(entropies)

        action = (opcode_id, reg_src_id, reg_dst_id, imm_id)
        out = {
            "action": action,
            "log_prob": total_log_prob,
            "entropy": total_entropy,
        }
        if return_logits:
            out["logits"] = all_logits
        return out

    def forward_log_prob(
        self,
        features: torch.Tensor,
        opcode_id: torch.Tensor,
        reg_src_id: torch.Tensor,
        reg_dst_id: torch.Tensor,
        imm_id: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward_logits(features, opcode_id, reg_src_id, reg_dst_id)
        op_lp, rs_lp, rd_lp, imm_lp = [
            F.log_softmax(lg, dim=-1) for lg in logits if lg is not None
        ]
        logp = (
            op_lp.gather(1, opcode_id.unsqueeze(-1)).squeeze(-1)
            + rs_lp.gather(1, reg_src_id.unsqueeze(-1)).squeeze(-1)
            + rd_lp.gather(1, reg_dst_id.unsqueeze(-1)).squeeze(-1)
            + imm_lp.gather(1, imm_id.unsqueeze(-1)).squeeze(-1)
        )
        return logp


# ---------------------------------------------------------------------------
# RLlib: custom action distribution
# ---------------------------------------------------------------------------

class TorchHierarchicalAutoregressiveDistribution(TorchDistributionWrapper):
    """RLlib action distribution for HierarchicalActionHead."""

    def __init__(self, inputs, model, **kwargs):
        super().__init__(inputs, model)
        self._action_logp = None

    @override(ActionDistribution)
    def sample(self):
        out = self.model.hierarchical_head.forward_sample(
            self.inputs, deterministic=False, return_logits=False
        )
        self._action_logp = out["log_prob"]
        return out["action"]

    @override(ActionDistribution)
    def deterministic_sample(self):
        out = self.model.hierarchical_head.forward_sample(
            self.inputs, deterministic=True, return_logits=False
        )
        self._action_logp = out["log_prob"]
        return out["action"]

    @override(ActionDistribution)
    def sampled_action_logp(self):
        return self._action_logp

    @override(ActionDistribution)
    def logp(self, actions):
        if isinstance(actions, (list, tuple)) and len(actions) == 4:
            o, rs, rd, imm = actions
        elif isinstance(actions, torch.Tensor) and actions.dim() == 2:
            o = actions[:, 0].long()
            rs = actions[:, 1].long()
            rd = actions[:, 2].long()
            imm = actions[:, 3].long()
        else:
            raise ValueError(f"Unexpected actions format: {type(actions)}")
        return self.model.hierarchical_head.forward_log_prob(
            self.inputs, o, rs, rd, imm
        )

    @override(ActionDistribution)
    def entropy(self):
        out = self.model.hierarchical_head.forward_sample(
            self.inputs, deterministic=False, return_logits=False
        )
        return out["entropy"]

    @override(ActionDistribution)
    def kl(self, other):
        return torch.zeros(self.inputs.shape[0], device=self.inputs.device)

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        cc = model_config.get("custom_model_config", model_config)
        return cc.get("hidden_dim", 256)


def register_hierarchical_action_dist():
    ModelCatalog.register_custom_action_dist(
        "TorchHierarchicalAutoregressiveDistribution",
        TorchHierarchicalAutoregressiveDistribution,
    )


# ---------------------------------------------------------------------------
# RLlib model
# ---------------------------------------------------------------------------

class SpecRLHierarchicalModel(TorchModelV2, nn.Module):
    """
    Uses HierarchicalActionHead: 4 heads, action = (opcode, reg_src, reg_dst, imm).
    Requires custom_action_dist=TorchHierarchicalAutoregressiveDistribution.
    """

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

        num_opcodes = get_num_opcodes() - 1
        num_regs = get_num_regs() - 1
        num_imms = get_num_imms()
        self.hierarchical_head = HierarchicalActionHead(
            input_dim=enc_dim,
            num_opcodes=num_opcodes,
            num_regs=num_regs,
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
            return _unflatten_obs(obs, self._seq_size, self._num_inputs), True
        return obs, False

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
        self._features = features
        return features, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None
        return self.value_head(self._features).squeeze(-1)


def register_specrl_hierarchical_model():
    ModelCatalog.register_custom_model("SpecRLHierarchicalModel", SpecRLHierarchicalModel)
    register_hierarchical_action_dist()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Demo: hierarchical action space, 4 heads."""
    action_space = get_hierarchical_action_space()
    print("Hierarchical action space:", action_space)
    print("  opcode:  Discrete(%d)" % action_space.spaces[0].n)
    print("  reg_src: Discrete(%d)" % action_space.spaces[1].n)
    print("  reg_dst: Discrete(%d)" % action_space.spaces[2].n)
    print("  imm:     Discrete(%d)" % action_space.spaces[3].n)

    head = HierarchicalActionHead(
        input_dim=256,
        num_opcodes=get_num_opcodes() - 1,
        num_regs=get_num_regs() - 1,
        num_imms=get_num_imms(),
        embed_dim=64,
        head_hidden=128,
    )
    features = torch.randn(4, 256)

    print("\n=== Sample (with logits) ===")
    out = head.forward_sample(features, deterministic=False, return_logits=True)
    o, rs, rd, imm = out["action"]
    for b in range(4):
        a = (int(o[b].item()), int(rs[b].item()), int(rd[b].item()), int(imm[b].item()))
        print(f"  action={a}, log_prob={out['log_prob'][b].item():.4f}")
    print("  logits keys:", list(out["logits"].keys()))

    print("\n=== Deterministic ===")
    out2 = head.forward_sample(features, deterministic=True, return_logits=False)
    for b in range(4):
        a = tuple(int(x[b].item()) for x in out2["action"])
        print(f"  action={a}")

    print("\n=== Log_prob of action ===")
    logp = head.forward_log_prob(features, o, rs, rd, imm)
    print("  log_prob:", logp.tolist())


if __name__ == "__main__":
    demo()
