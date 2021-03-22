"""
unimoco model definition, which is inherited from moco.
"""
from .moco import MoCo, concat_all_gather
import torch
from torch import nn

class UniMoCo(MoCo):
    """
    build a UniMoCo model with the same hyper-parameter with MoCo
    """
    def __init__(self, *args, **kwargs):
        """check moco.py for more arguments details.
        """
        super().__init__(*args, **kwargs)
        # initialize a label queue with shape K.
        # all the label is -1 by default.
        self.register_buffer("label_queue", torch.zeros(self.K).long() - 1)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys and labels before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        batch_size = keys.shape[0]
        # print(batch_size)
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
 
        # replace the keys and labels at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T # this queue is feature queue
        self.label_queue[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
    
    def forward(self, im_q, im_k, labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            labels: a batch of label for images (-1 for unsupervised images)
        Output:
            logits: with shape Nx(1+K)
            targets: with shape Nx(1+K)
            fake_targets: to report the top-1, top-5 accuracy as MoCo,
                          it returns the index of the data augmented image.
        """
        ####################code from MoCo####################
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        ####################above code is same as MoCo####################

        batch_size = labels.shape[0]
        # one-hot target from augmented image
        positive_target = torch.ones((batch_size, 1)).cuda()
        # find same label images from label queue
        # for the query with -1, all 
        targets = ((labels[:, None] == self.label_queue[None, :]) & (labels[:, None] != -1)).float().cuda()
        targets = torch.cat([positive_target, targets], dim=1)
        # dequeue and enqueue
        self._dequeue_and_enqueue(k, labels)

        return logits, targets, torch.zeros(logits.shape[0], dtype=torch.long).cuda()  # to report the top-1, top-5
