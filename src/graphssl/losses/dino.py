"""DINO multi-view cross-entropy loss."""

from torch import nn, Tensor


class DINOLoss(nn.Module):
    """Cross-entropy between teacher softmax and student log-softmax over N views.

    Inputs are pre-processed by DINOHead.forward():
    - student_out: log-softmax with student_temp, shape [n_views * batch, D]
    - teacher_out: softmax with center subtraction and teacher_temp,
                   shape [n_global_views * batch, D]

    For every (teacher_i, student_j) pair with i != j, computes H(teacher_i, student_j)
    and averages over all such pairs. When n_global_views == n_views == 2 this
    reduces to −0.5 * ((t2 * s1).sum(−1).mean() + (t1 * s2).sum(−1).mean()).
    """

    def forward(
        self,
        student_out: Tensor,
        teacher_out: Tensor,
        n_global_views: int,
        n_views: int,
    ) -> Tensor:
        student_chunks = student_out.chunk(n_views)
        teacher_chunks = teacher_out.chunk(n_global_views)

        total_loss = 0.0
        n_terms = 0
        for iq, q in enumerate(teacher_chunks):
            for iv, s in enumerate(student_chunks):
                if iv == iq:
                    continue
                total_loss += (-q * s).sum(dim=-1).mean()
                n_terms += 1

        return total_loss / max(n_terms, 1)
