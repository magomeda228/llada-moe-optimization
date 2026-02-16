import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
import warnings
import transformers

class FastLLaDAMoE(nn.Module):
    def __init__(self, original_block):
        super().__init__()
        self.num_experts = original_block.num_experts
        self.top_k = original_block.top_k
        self.hidden_dim = original_block.gate.in_features

        #copy router
        self.gate = original_block.gate
        self.expert_bias = original_block.expert_bias

        #stack weights
        experts = original_block.experts

        #w_gate_up: [E, 2*Inter, Hidden]
        w_gate = torch.stack([e.gate_proj.weight.data for e in experts], dim=0)
        w_up = torch.stack([e.up_proj.weight.data for e in experts], dim=0)
        self.w_gate_up = nn.Parameter(torch.cat([w_gate, w_up], dim=1))

        #w_down
        self.w_down = nn.Parameter(torch.stack([e.down_proj.weight.data for e in experts], dim=0))

        self.inter_dim = experts[0].gate_proj.out_features

    def forward(self, hidden_states):
        torch.cuda.nvtx.range_push("ROUTER")
        bsz, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)

        #router
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        if self.expert_bias is not None:
            routing_weights = routing_weights + self.expert_bias
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("SORT_AND_SPLIT")

        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights.to(x.dtype)

        #sort_indeces
        expert_ids = selected_experts.view(-1)
        flat_weights = routing_weights.view(-1)

        sorted_expert_ids, sorted_indices = torch.sort(expert_ids)
        token_indices = sorted_indices // self.top_k

        #tokens in right order
        x_sorted = x[token_indices].contiguous()

        #counts and split experts
        expert_counts = torch.bincount(expert_ids, minlength=self.num_experts)
        counts_list = expert_counts.tolist()

        #cut input for each expert
        inputs_split = torch.split(x_sorted, counts_list)

        #buffer for output
        final_output = torch.zeros_like(x)

        #loop over experts and compute outputs
        current_idx = 0
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("EXPERTS_LOOP")
        for i, (inp, count) in enumerate(zip(inputs_split, counts_list)):
            if count == 0:
                continue

            w1 = self.w_gate_up[i]
            gate_up = F.linear(inp, w1)
            gate, up = gate_up.chunk(2, dim=1)
            inter = F.silu(gate) * up

            w2 = self.w_down[i]
            out = F.linear(inter, w2)

            #Indeces of tokens for this expert
            chunk_indices = token_indices[current_idx : current_idx + count]

            #Weights for this expert
            chunk_weights = flat_weights[sorted_indices[current_idx : current_idx + count]].unsqueeze(1)

            #atomic add
            final_output.index_add_(0, chunk_indices, out * chunk_weights)

            #Shift
            current_idx += count
        torch.cuda.nvtx.range_pop()
        return final_output.view(bsz, seq_len, hidden_dim)


def optimize_llada_moe(model):

    required_version = "4.57.6"
    current_version = transformers.__version__
    
    if version.parse(current_version) != version.parse(required_version):
        warnings.warn(
            f"Warning: This optimization is tested on transformers=={required_version}. "
            f"You are using {current_version}. Use at your own risk."
        )

    print("Moe optimization started...")
    count = 0

    for name, module in model.named_modules():
        if module.__class__.__name__ == 'LLaDAMoESparseMoeBlock':
            fast_block = FastLLaDAMoE(module).to(module.gate.weight.device).to(module.gate.weight.dtype)
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            if parent_name:
                parent = model.get_submodule(parent_name)
                child_name = name.rsplit('.', 1)[1]
            else:
                parent = model
                child_name = name

            setattr(parent, child_name, fast_block)
            count += 1
    print(f"Count: {count}")
    print("Forward was patched. Moe optimization finished.")

    return model