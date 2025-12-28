import torch
import numpy as np

class RetMetric(object):
    def __init__(self, feats, labels):
        if len(feats) == 2 and type(feats) == list:
            self.is_equal_query = False
            self.gallery_feats, self.query_feats = feats
            self.gallery_labels, self.query_labels = labels
        else:
            self.is_equal_query = True
            self.gallery_feats = self.query_feats = feats
            self.gallery_labels = self.query_labels = labels
        self.sim_mat = np.matmul(self.query_feats, np.transpose(self.gallery_feats))

    def recall_k(self, k=1):
        m = len(self.sim_mat)
        match_counter = 0
        for i in range(m):
            pos_sim = self.sim_mat[i][self.gallery_labels == self.query_labels[i]]
            neg_sim = self.sim_mat[i][self.gallery_labels != self.query_labels[i]]
            thresh = np.sort(pos_sim)[-2] if self.is_equal_query else np.max(pos_sim)
            if np.sum(neg_sim > thresh) < k:
                match_counter += 1
        return float(match_counter) / m
    
def compute_recall_at_k(embeddings, targets, k_values=(1, 2, 4)):
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    feats_np = embeddings.cpu().numpy()
    labels_np = targets.cpu().numpy()
    ret_metric = RetMetric(feats_np, labels_np)
    recalls = {}
    for k in k_values:
        recalls[f"R@{k}"] = ret_metric.recall_k(k=k)
    return recalls