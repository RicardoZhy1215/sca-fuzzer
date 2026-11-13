"""
CLAP-asm assembly encoder (Wang et al., "CLAP: Learning Transferable Binary Code Representations with
Natural Language Supervision", ISSTA'24; weights: huggingface.co/hustcw/clap-asm, MIT licence).

The model and tokenizer classes below are copied from the repository's clap_modeling.py (MIT,
Copyright (c) 2024 Hustcw), so the weights (a safetensors file) load without trust_remote_code.
Only the asm encoder is kept.

  enc = ClapAsm()                       # from the local HF cache
  v = enc.embed([["mov rax, qword ptr [r14 + rbx]", "mulsd xmm0, xmm1"], ...])   # (n, 768), unit norm
"""
import glob
import os

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BatchEncoding, MPNetTokenizerFast
from transformers.models.roformer.modeling_roformer import (RoFormerAttention, RoFormerEmbeddings,
                                                            RoFormerEncoder, RoFormerIntermediate, RoFormerLayer,
                                                            RoFormerModel, RoFormerOutput, RoFormerPreTrainedModel,
                                                            RoFormerSelfAttention)

SNAP = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--hustcw--clap-asm/snapshots/*"))


class AsmTokenizer(MPNetTokenizerFast):
    """A function is a dict {"1": instruction, "2": ...}; every token also carries the index of its
    instruction (token type INSTR<k>)."""

    @property
    def pad_token_type_id(self) -> int:
        return self.pad_token_id

    def tokenize_function(self, function):
        total_len = 0
        out = {"token": [], "instr": []}
        for key, value in function.items():
            tokens = self.tokenize(value.replace(',', ''), max_length=20, truncation=True, add_special_tokens=False)
            out["token"].extend(tokens)
            out["instr"].extend(["INSTR" + key] * len(tokens))
            total_len += len(tokens)
            if total_len > self.model_max_length:
                out["token"] = out["token"][:self.model_max_length]
                out["instr"] = out["instr"][:self.model_max_length]
                break
        return out

    def __call__(self, functions, **kwargs):
        res = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
        for function in functions:
            t = self.tokenize_function(function)
            ids = self.convert_tokens_to_ids(t["token"])
            res["input_ids"].append(ids)
            res["attention_mask"].append([1] * len(ids))
            res["token_type_ids"].append(self.convert_tokens_to_ids(t["instr"]))
        return self.pad(BatchEncoding(res), **kwargs)


class JRoFormerEmbeddings(RoFormerEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = self.word_embeddings


class JRoFormerSelfAttention(RoFormerSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)


class JRoFormerAttention(RoFormerAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = JRoFormerSelfAttention(config)


class JRoFormerLayer(RoFormerLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = JRoFormerAttention(config)
        self.intermediate = RoFormerIntermediate(config)
        self.output = RoFormerOutput(config)


class JRoFormerEncoder(RoFormerEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([JRoFormerLayer(config) for _ in range(config.num_hidden_layers)])


class JRoFormerModel(RoFormerModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = JRoFormerEmbeddings(config)
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)
        self.encoder = JRoFormerEncoder(config)
        self.post_init()


class AsmEncoder(RoFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.jroformer = JRoFormerModel(config)
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        h = self.jroformer(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return F.normalize(self.projection(v), p=2, dim=1)


class ClapAsm:
    def __init__(self, device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        from safetensors.torch import load_file
        from transformers import RoFormerConfig
        self.tok = AsmTokenizer.from_pretrained(SNAP[0], local_files_only=True)
        # built from the config and loaded by hand: transformers 5's from_pretrained does not handle the
        # token-type embedding being the word embedding (it reports it missing and then fails)
        self.enc = AsmEncoder(RoFormerConfig.from_pretrained(SNAP[0]))
        sd = load_file(os.path.join(SNAP[0], "model.safetensors"))
        missing, unexpected = self.enc.load_state_dict(sd, strict=False)
        assert set(missing) <= {"jroformer.embeddings.token_type_embeddings.weight"} and not unexpected, \
            (missing, unexpected)
        assert self.enc.jroformer.embeddings.token_type_embeddings is self.enc.jroformer.embeddings.word_embeddings
        self.enc = self.enc.to(self.dev).eval()

    @torch.no_grad()
    def embed(self, programs, batch=32):
        out = []
        for k in range(0, len(programs), batch):
            fs = [{str(i + 1): ins for i, ins in enumerate(p[:1024])} for p in programs[k:k + batch]]
            b = self.tok(fs, padding=True, return_tensors="pt").to(self.dev)
            out.append(self.enc(**b).float().cpu())
        return torch.cat(out).numpy()
