"""
A nice and efficient mixed AdamW/Muon Combined Optimizer.
Usually the embeddings and scalars go into AdamW, and the matrix parameters go into Muon.
The same class handles both single GPU and distributed training: when there is no
multi-rank process group, the communication ops are simply skipped and every rank
(i.e. the only rank) owns all of the parameters.

Adapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
"""

import torch
import torch.distributed as dist
from torch import Tensor
from nanochat.common import COMPUTE_DTYPE

# -----------------------------------------------------------------------------
"""
Good old AdamW optimizer, fused kernel.
https://arxiv.org/abs/1711.05101
"""

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,              # (32768, 768) - parameter tensor
    grad: Tensor,           # (32768, 768) - gradient, same shape as p
    exp_avg: Tensor,        # (32768, 768) - first moment, same shape as p
    exp_avg_sq: Tensor,     # (32768, 768) - second moment, same shape as p
    step_t: Tensor,         # () - 0-D CPU tensor, step count
    lr_t: Tensor,           # () - 0-D CPU tensor, learning rate
    beta1_t: Tensor,        # () - 0-D CPU tensor, beta1
    beta2_t: Tensor,        # () - 0-D CPU tensor, beta2
    eps_t: Tensor,          # () - 0-D CPU tensor, epsilon
    wd_t: Tensor,           # () - 0-D CPU tensor, weight decay
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
    All in one compiled graph to eliminate Python overhead between ops.
    The 0-D CPU tensors avoid recompilation when hyperparameter values change.
    """
    # Some params (wte, value_embeds) are stored in bf16, so do the math in fp32 and
    # cast back at the end. MPS errors on mixed-dtype ops (CUDA promotes them), and
    # scalar arithmetic like 1 - beta2 loses all precision in bf16. compile fuses the casts.
    p32 = p.float()
    exp_avg32 = exp_avg.float()
    exp_avg_sq32 = exp_avg_sq.float()
    grad32 = grad.float()
    # Weight decay (decoupled, applied before the update)
    p32.mul_(1 - lr_t * wd_t)
    # Update running averages (lerp_ is cleaner and fuses well)
    exp_avg32.lerp_(grad32, 1 - beta1_t)
    exp_avg_sq32.lerp_(grad32.square(), 1 - beta2_t)
    # Bias corrections
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    # Compute update and apply
    denom = (exp_avg_sq32 / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p32.add_(exp_avg32 / denom, alpha=-step_size)
    # Write back (no-ops in the common case where everything is already fp32)
    p.copy_(p32)
    exp_avg.copy_(exp_avg32)
    exp_avg_sq.copy_(exp_avg_sq32)

# -----------------------------------------------------------------------------
"""
Muon optimizer adapted and simplified from modded-nanogpt.
https://github.com/KellerJordan/modded-nanogpt

Background:
Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
zero even beyond the point where the iteration no longer converges all the way to one everywhere
on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
performance at all relative to UV^T, where USV^T = G is the SVD.

Here, an alternative to Newton-Schulz iteration with potentially better convergence properties:
Polar Express Sign Method for orthogonalization.
https://arxiv.org/pdf/2505.16932
by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

NorMuon variance reduction: per-neuron/column adaptive learning rate that normalizes
update scales after orthogonalization (Muon's output has non-uniform scales across neurons).
https://arxiv.org/pdf/2510.05491

Two more (very) slight and optional improvements:
1) MuonEq row equilibration: rescale each row to the mean row norm so the spectrum
entering orthogonalization is better conditioned (https://arxiv.org/abs/2603.28254)
2) Muon+ renormalization: snap the Frobenius norm to sqrt(min(m, n)), the norm of an exactly
semi-orthogonal matrix, correcting for under-convergence of the polar iteration (https://arxiv.org/abs/2602.21545)

Some of the changes in nanochat implementation:
- Uses a simpler, more general approach to parameter grouping and stacking
- Uses a single fused kernel for the momentum -> polar_express -> variance_reduction -> update step
- Makes no assumptions about model architecture (e.g. that attention weights are fused into QKVO format)
"""

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,          # (12, 768, 3072) - stacked gradients
    stacked_params: Tensor,         # (12, 768, 3072) - stacked parameters
    momentum_buffer: Tensor,        # (12, 768, 3072) - first moment buffer
    second_momentum_buffer: Tensor, # (12, 768, 1) or (12, 1, 3072) - factored second moment
    momentum_t: Tensor,             # () - 0-D CPU tensor, momentum coefficient
    lr_t: Tensor,                   # () - 0-D CPU tensor, learning rate
    wd_t: Tensor,                   # () - 0-D CPU tensor, weight decay
    beta2_t: Tensor,                # () - 0-D CPU tensor, beta2 for second moment
    ns_steps: int,                  # 5 - number of Newton-Schulz/Polar Express iterations
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
) -> None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    All in one compiled graph to eliminate Python overhead between ops.
    Some of the constants are 0-D CPU tensors to avoid recompilation when values change.
    """

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Cast to bf16 for speed when available; skip cast otherwise (fp16 is unstable here due to limited exponent range)
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g

    # MuonEq row equilibration: rescale each row to the mean row norm so the spectrum entering orthogonalization is better conditioned
    target = X.float().norm(dim=(-2, -1), keepdim=True) / (X.size(-2) ** 0.5)
    row_norm = X.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    X = X * (target / row_norm).to(X.dtype)

    # Polar Express orthogonalization: replace each update with the nearest orthogonal matrix
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1): # Tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else: # Wide matrix (original math)
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    # Cast back to the param dtype (MPS errors on the mixed-dtype ops below when X is bf16)
    g = X.to(stacked_params.dtype)

    # Muon+ renormalization: snap Frobenius norm to sqrt(min(m, n))
    target_norm = min(g.size(-2), g.size(-1)) ** 0.5
    current_norm = g.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    g = g * (target_norm / current_norm).to(g.dtype)

    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

# -----------------------------------------------------------------------------


class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others, built around
    flat parameter/gradient tapes ("flat theta").

    At construction, every parameter is re-housed as a view into one of a few large
    contiguous tapes (p.data = tape_view), and every p.grad is pre-bound as a view
    into a matching flat gradient tape, so autograd accumulates gradients directly
    into the tapes. One optimizer step is then just a handful of tape-level
    collectives instead of one or two per parameter:

        Muon tape (one, fp32):     reduce_scatter(grad) -> update own chunk -> all_gather(theta)
        AdamW tapes (one/dtype):   reduce_scatter(grad) -> update own chunk -> all_gather(theta)

    All collectives run in place (NCCL's documented in-place semantics: the chunk
    is a view into the tape at rank*chunk), so there is no torch.stack on the way
    in and no copy-back on the way out - the params ARE the tape.

    There are multiple tapes, not one, because dtypes can't mix in a flat tensor
    (embeddings are bf16, matrices fp32) and because Muon and AdamW shard on
    different atoms:

    - Muon's atom is a whole matrix (it cannot orthogonalize part of one), so the
      Muon tape is laid out rank-major: [rank0's bundle | rank1's bundle | ...],
      each bundle holding the same number of whole matrices of every "size class".
      A single equal-chunk reduce_scatter then hands every rank contiguous runs of
      whole matrices that view directly as (K, m, n) stacks for the fused kernel.
      Groups whose matrices have equal numel (e.g. mlp c_fc (4n, n) and c_proj
      (n, 4n)) pool into one size class, which makes the per-rank count come out
      even where per-shape counts would not - no zero-padding at any default depth.
      Leftover blocks are padded with all-zero dummy matrices (Muon's update of a
      zero matrix is zero, so they ride along harmlessly).

    - AdamW's atom is an element, so its tapes are laid out param-major and the
      equal chunks may cut anywhere - mid-row, mid-tensor. Each rank applies the
      right group hyperparameters to the sub-segments of its chunk (ZeRO-2:
      exp_avg/exp_avg_sq exist only for the local chunk).

    The tape layout depends on world_size, so it is a runtime detail of the
    optimizer: checkpoints save the ordinary name -> tensor dict (the model) and
    this rank's chunk-sized optimizer state; resuming requires the same world_size.

    On a single rank (no process group) all communication is skipped and the
    "chunk" is the whole tape - Muon still gets stack-free (K, m, n) views.

    The p.grad tape aliasing is an optimization, not a correctness requirement:
    step() re-adopts any gradient that was severed from the tape (e.g. by an
    external p.grad assignment) before reducing. Use this optimizer's zero_grad(),
    which zeroes the tapes and preserves the aliasing.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
        All params in a Muon group must have the same shape (caller's responsibility).
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        assert all(g['kind'] in ('adamw', 'muon') for g in self.param_groups), "unknown optimizer kind"
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # The tape layout is a function of world size, decided once at construction
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self._grad_views = {}  # param -> its pre-bound view into a grad tape
        self._grad_tapes = []
        self._build_muon_tape()
        self._build_adamw_tapes()

    def _build_muon_tape(self):
        """Lay out all Muon params in one rank-major tape of whole matrices."""
        W, r = self.world_size, self.rank
        # Note: groups are referenced by INDEX into self.param_groups, never by dict
        # reference - load_state_dict() replaces the group dicts, and the scheduler
        # mutates the new ones, so a captured reference would read stale hyperparams.
        muon_groups = [(gi, g) for gi, g in enumerate(self.param_groups) if g['kind'] == 'muon']
        self._muon = None
        if not muon_groups:
            return
        for _, group in muon_groups:
            shapes = {p.shape for p in group['params']}
            assert len(shapes) == 1, "all params in a Muon group must share one shape"

        # Partition the matrices into size classes by per-matrix numel. Same-numel
        # groups pool their counts, so the blocks divide more evenly across ranks.
        classes = {}  # per-matrix numel -> list of (group_idx, param) in group order
        for gi, group in muon_groups:
            for p in group['params']:
                classes.setdefault(p.numel(), []).append((gi, p))

        # Bundle layout: every rank's bundle holds k_c whole matrices of class c
        # (k_c = ceil(K_c / W); the last slots may be all-zero dummy blocks)
        bundle = 0
        class_infos = []
        for numel, blocks in classes.items():
            k = -(-len(blocks) // W)  # ceil division
            class_infos.append(dict(numel=numel, blocks=blocks, K=len(blocks), k=k, class_offset=bundle))
            bundle += k * numel

        p0 = muon_groups[0][1]['params'][0]
        dtype, device = p0.dtype, p0.device
        assert all(p.dtype == dtype for _, g in muon_groups for p in g['params']), "Muon params must share one dtype"
        theta = torch.zeros(W * bundle, dtype=dtype, device=device)  # zeros: dummy blocks stay zero forever
        grad_tape = torch.zeros_like(theta)

        # Re-house every param and its grad as views into the tapes
        for ci in class_infos:
            for j, (gi, p) in enumerate(ci['blocks']):
                owner, slot = divmod(j, ci['k'])
                offset = owner * bundle + ci['class_offset'] + slot * ci['numel']
                theta_view = theta[offset:offset + ci['numel']].view_as(p)
                theta_view.copy_(p.detach())
                p.data = theta_view  # p now aliases the tape (same Python object everywhere)
                p.grad = grad_tape[offset:offset + ci['numel']].view_as(p)
                self._grad_views[p] = p.grad

        # This rank's compute plan: contiguous per-group runs of whole matrices
        # inside our bundle, each viewable as a (count, m, n) stack. Dummy padding
        # blocks are attached to the last real run of their class.
        subruns = []  # dicts: gidx (into param_groups), count, start (absolute tape offset), shape
        for ci in class_infos:
            runs = []  # [group_idx, count]
            for j in range(r * ci['k'], (r + 1) * ci['k']):
                gi = ci['blocks'][min(j, ci['K'] - 1)][0]  # dummies -> last real block's group
                if runs and runs[-1][0] == gi:
                    runs[-1][1] += 1
                else:
                    runs.append([gi, 1])
            slot = 0
            for gi, count in runs:
                start = r * bundle + ci['class_offset'] + slot * ci['numel']
                shape = self.param_groups[gi]['params'][0].shape
                subruns.append(dict(gidx=gi, count=count, start=start, shape=shape, numel=count * ci['numel']))
                slot += count

        # Persistent optimizer state, sharded: only this rank's shard exists here.
        # Keyed under an anchor param so torch's state_dict/load_state_dict work.
        second_momentum = []
        for sr in subruns:
            m, n = sr['shape'][-2], sr['shape'][-1]
            state_shape = (sr['count'], m, 1) if m >= n else (sr['count'], 1, n)
            second_momentum.append(torch.zeros(state_shape, dtype=dtype, device=device))
        self.state[p0] = dict(
            momentum_tape=torch.zeros(bundle, dtype=dtype, device=device),
            second_momentum=second_momentum,
        )
        self._grad_tapes.append(grad_tape)
        # a rank's bundle is exactly its collective chunk, so store it as 'chunk' like the AdamW tapes
        self._muon = dict(theta=theta, grad_tape=grad_tape, chunk=bundle, subruns=subruns, anchor=p0)

    def _build_adamw_tapes(self):
        """Lay out AdamW params in one param-major tape per dtype, sharded elementwise."""
        W, r = self.world_size, self.rank
        self._adamw_tapes = []
        by_dtype = {}  # dtype -> list of (group_idx, param) in group order
        for gi, group in enumerate(self.param_groups):
            if group['kind'] != 'adamw':
                continue
            for p in group['params']:
                by_dtype.setdefault(p.dtype, []).append((gi, p))
        for dtype, blocks in by_dtype.items():
            device = blocks[0][1].device
            total = sum(p.numel() for _, p in blocks)
            padded = -(-total // W) * W  # pad with a few zero dummy elements so W divides the tape

            theta = torch.zeros(padded, dtype=dtype, device=device)
            grad_tape = torch.zeros_like(theta)
            offset = 0
            intervals = []  # (group_idx, start, end) per param, in tape order
            for gi, p in blocks:
                theta_view = theta[offset:offset + p.numel()].view_as(p)
                theta_view.copy_(p.detach())
                p.data = theta_view
                p.grad = grad_tape[offset:offset + p.numel()].view_as(p)
                self._grad_views[p] = p.grad
                intervals.append((gi, offset, offset + p.numel()))
                offset += p.numel()

            # This rank's segments: our chunk intersected with each param's interval,
            # merged where neighbors share a group. AdamW is elementwise, so the
            # chunk cuts may fall anywhere - mid-row, mid-tensor.
            chunk = padded // W
            chunk_start, chunk_end = r * chunk, (r + 1) * chunk
            segments = []  # [group_idx, start, end] in absolute tape offsets
            for gi, a, b in intervals:
                a, b = max(a, chunk_start), min(b, chunk_end)
                if a >= b:
                    continue
                if segments and segments[-1][0] == gi and segments[-1][2] == a:
                    segments[-1][2] = b
                else:
                    segments.append([gi, a, b])

            anchor = blocks[0][1]
            self.state[anchor] = dict(
                step=0,
                exp_avg=torch.zeros(chunk, dtype=dtype, device=device),
                exp_avg_sq=torch.zeros(chunk, dtype=dtype, device=device),
            )
            self._grad_tapes.append(grad_tape)
            self._adamw_tapes.append(dict(theta=theta, grad_tape=grad_tape, chunk=chunk, segments=segments, anchor=anchor))

    def _ensure_grad_views(self):
        """
        Normally autograd accumulates directly into the grad tapes because every
        p.grad was pre-bound as a tape view at construction. If the aliasing was
        severed (external p.grad assignment or set_to_none), adopt the values into
        the tape and restore the view so the reduce always sees the real gradients.
        """
        for p, view in self._grad_views.items():
            grad = p.grad
            if grad is None:
                view.zero_()
            elif grad.data_ptr() != view.data_ptr():
                view.copy_(grad)
            p.grad = view

    def zero_grad(self, set_to_none=True):
        """Zero the flat grad tapes (a few big memsets) and keep the p.grad aliasing."""
        for tape in self._grad_tapes:
            tape.zero_()
        for p, view in self._grad_views.items():
            p.grad = view

    def _step_muon(self):
        """Muon updates for the matrices this rank owns, in place in the param tape."""
        t = self._muon
        state = self.state[t['anchor']]
        momentum_tape = state['momentum_tape']
        bundle_start = self.rank * t['chunk']
        for i, sr in enumerate(t['subruns']):
            group = self.param_groups[sr['gidx']] # by index: load_state_dict replaces the group dicts
            count, shape = sr['count'], sr['shape']
            start, numel = sr['start'], sr['numel']
            grads = t['grad_tape'][start:start + numel].view(count, *shape)
            params_stack = t['theta'][start:start + numel].view(count, *shape)
            momentum = momentum_tape[start - bundle_start:start - bundle_start + numel].view(count, *shape)
            red_dim = -1 if shape[-2] >= shape[-1] else -2
            self._muon_momentum_t.fill_(group['momentum'])
            self._muon_beta2_t.fill_(group['beta2'])
            self._muon_lr_t.fill_(group['lr'] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
            self._muon_wd_t.fill_(group['weight_decay'])
            muon_step_fused(
                grads, params_stack, momentum, state['second_momentum'][i],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                group['ns_steps'], red_dim,
            )

    def _step_adamw(self, t):
        """AdamW updates for this rank's chunk of one tape, segment by segment."""
        state = self.state[t['anchor']]
        state['step'] += 1
        self._adamw_step_t.fill_(state['step'])
        chunk_start = self.rank * t['chunk']
        for gi, a, b in t['segments']:
            group = self.param_groups[gi] # by index: load_state_dict replaces the group dicts
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(
                t['theta'][a:b], t['grad_tape'][a:b],
                state['exp_avg'][a - chunk_start:b - chunk_start],
                state['exp_avg_sq'][a - chunk_start:b - chunk_start],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

    @torch.no_grad()
    def step(self):
        W, r = self.world_size, self.rank
        self._ensure_grad_views()

        # Phase 1: launch all async reduce_scatters, in place (output chunk is a view
        # into the grad tape). NCCL executes them in launch order, so the Muon tape -
        # whose update compute is the heaviest - goes first.
        tapes = ([self._muon] if self._muon is not None else []) + self._adamw_tapes
        reduces = []
        if W > 1:
            for t in tapes:
                chunk = t['chunk']
                grad_chunk = t['grad_tape'][r * chunk:(r + 1) * chunk]
                work = dist.reduce_scatter_tensor(grad_chunk, t['grad_tape'], op=dist.ReduceOp.AVG, async_op=True)
                reduces.append(work)

        # Phase 2: as each reduce lands, compute this rank's updates in place in the
        # param tape, then launch the in-place all_gather that rebroadcasts them.
        gathers = []
        for i, t in enumerate(tapes):
            if W > 1:
                reduces[i].wait()
            if t is self._muon:
                self._step_muon()
            else:
                self._step_adamw(t)
            if W > 1:
                chunk = t['chunk']
                theta_chunk = t['theta'][r * chunk:(r + 1) * chunk]
                work = dist.all_gather_into_tensor(t['theta'], theta_chunk, async_op=True)
                gathers.append(work)

        # Phase 3: wait for the gathers. Params are views into the tapes, so there
        # is nothing to copy back.
        for work in gathers:
            work.wait()
