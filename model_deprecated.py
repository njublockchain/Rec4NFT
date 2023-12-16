# %%
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import ModelType, init_logger, init_seed, set_color
from recbole.trainer import Trainer
from recbole.model.abstract_recommender import AbstractRecommender
from tqdm import tqdm

logger = getLogger()


# configurations initialization
class SequentialRecommender(AbstractRecommender):
    """
    This is a abstract sequential recommender. All the sequential model should implement This class.
    """

    type = ModelType.SEQUENTIAL

    def __init__(self, config, dataset):
        super(SequentialRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.ITEM_SEQ = self.ITEM_ID + config["LIST_SUFFIX"]
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.POS_ITEM_ID = self.ITEM_ID
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.n_items = dataset.num(self.ITEM_ID)

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the spexific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)


config = Config(
    # model=RecNFT,
    model=SequentialRecommender,
    dataset="nft",
    config_file_list=[
        "./props/overall.yaml",
        "./props/nft.yaml",
    ],
)
dataset = create_dataset(config)
logger.info(dataset)

# dataset splitting
train_data, valid_data, test_data = data_preparation(config, dataset)

# %%
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.layers import TransformerEncoder
from torch.distributions import Normal, MixtureSameFamily, Categorical
from sklearn.mixture import GaussianMixture


class TransNet(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["embedding_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]

        self.position_embedding = nn.Embedding(
            dataset.field2seqlen[config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]],
            self.hidden_size,
        )
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.fn = nn.Linear(self.hidden_size, 1)

        self.apply(self._init_weights)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def forward(self, item_seq, item_emb, collection_emb, value_emb):
        mask = item_seq.gt(0)

        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = torch.cat(
            [position_embedding + item_emb, collection_emb, value_emb], dim=1
        )
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        extended_attention_mask = F.pad(extended_attention_mask, (0, 2, 0, 2))

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]

        alpha = self.fn(output).to(torch.double)
        mask = F.pad(mask, (0, 2)).unsqueeze(-1)
        alpha = torch.where(mask, alpha, -9e15)
        alpha = torch.softmax(alpha, dim=1, dtype=torch.float)
        return alpha

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class GMMLayer(nn.Module):
    def __init__(self, num_components, num_features):
        super(GMMLayer, self).__init__()
        self.num_components = num_components
        self.num_features = num_features

        # initialize means, std_devs, and weights
        self.means = nn.Parameter(torch.randn(num_components, num_features))
        self.std_devs = nn.Parameter(torch.rand(num_components, num_features))
        self.weights = nn.Parameter(torch.randn(num_components))

    def forward(self, x):
        # create mixture distribution
        mixture_distribution = Categorical(logits=self.weights)
        component_distribution = Normal(self.means, self.std_devs.abs())
        gmm = MixtureSameFamily(mixture_distribution, component_distribution)

        # calulate log likelihood for training
        log_prob = gmm.log_prob(x)
        return log_prob

class RecNFT(SequentialRecommender):
    r"""CORE is a simple and effective framewor, which unifies the representation spac
    for both the encoding and decoding processes in session-based recommendation.
    """

    def __init__(self, config, dataset, n_collections):
        super(RecNFT, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.loss_type = config["loss_type"]

        self.n_collections = n_collections
        self.collection_embedding = nn.Embedding(
            self.n_collections, self.embedding_size, padding_idx=0
        )

        self.value_dim = 2  # = bundleValue + bundleCount
        mlp = nn.Sequential(
            nn.Linear(self.value_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.embedding_size),
        )
        self.value_embedding = mlp

        self.sess_dropout = nn.Dropout(config["sess_dropout"])
        self.item_dropout = nn.Dropout(config["item_dropout"])
        self.temperature = config["temperature"]

        # item embedding
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )

        # Clustering
        n_components = 2
        n_features = 2
        self.gmm = GMMLayer(n_components, n_features)
        
        # Attn
        self.net = TransNet(config, dataset)

        if self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE']!")

        # parameters initialization
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, item_seq, label, collection_id, bundle_value, bundle_count):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.sess_dropout(item_emb)

        collection_emb = self.collection_embedding(collection_id)
        collection_emb = collection_emb.unsqueeze(1)

        value = torch.cat([bundle_value.unsqueeze(1), bundle_count.unsqueeze(1)], dim=1)
        value_emb = self.value_embedding(value)
        value_emb = value_emb.unsqueeze(1)

        input_emb = torch.cat([item_emb, collection_emb, value_emb], dim=1)

        # 调用 TransNet
        alpha = self.net(
            item_seq, item_emb, collection_emb, value_emb
        )  # 将 collection_id 和 value 传递给 TransNet
        # seq_output = torch.sum(alpha * item_emb, dim=1)
        seq_output = torch.sum(alpha * input_emb, dim=1)
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output

    def calculate_loss(self, interaction):
        # user = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        pos_items = interaction[self.POS_ITEM_ID]

        # interaction:
        # ['user_id', 'item_id', 'collection_id',
        #  'rating', 'timestamp', 'bundle_value',
        #  'bundle_count', 'item_length', 'item_id_list',
        #  'collection_id_list', 'rating_list',
        #  'timestamp_list', 'bundle_value_list',
        #  'bundle_count_list']

        # X = torch.cat([
        #     torch.mean(interaction["bundle_value_list"], dim=1, keepdim=True),
        #     torch.mean(interaction["bundle_count_list"], dim=1, keepdim=True),
        # ], dim=1).cpu().numpy()
        # self.gmm.fit(X)
        # label = self.gmm.predict(X)
        # label = torch.tensor(label).to(torch.long).to(config["device"])

        # X = torch.cat(
        #     [
        #         torch.mean(interaction["bundle_value_list"], dim=1, keepdim=True),
        #         torch.mean(interaction["bundle_count_list"], dim=1, keepdim=True),
        #     ],
        #     dim=1,
        # )
        X = torch.stack(
            [
                interaction["bundle_value_list"],
                interaction["bundle_count_list"],
            ],
            dim=2,
        )
        label = self.gmm(X)

        seq_output = self.forward(
            item_seq,
            label,
            interaction["collection_id"],
            interaction["bundle_value"],
            interaction["bundle_count"],
        )

        all_item_emb = self.item_embedding.weight
        all_item_emb = self.item_dropout(all_item_emb)
        # Robust Distance Measuring (RDM)
        all_item_emb = F.normalize(all_item_emb, dim=-1)
        logits = (
            torch.matmul(seq_output, all_item_emb.transpose(0, 1)) / self.temperature
        )
        loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        # X = torch.cat([
        #     torch.mean(interaction["bundle_value_list"], dim=1, keepdim=True),
        #     torch.mean(interaction["bundle_count_list"], dim=1, keepdim=True),
        # ], dim=1).cpu().numpy()
        # self.gmm.fit(X)
        # label = self.gmm.predict(X)
        # label = torch.tensor(label).to(torch.long).to(config["device"])

        X = torch.stack(
            [
                interaction["bundle_value_list"],
                interaction["bundle_count_list"],
            ],
            dim=2,
        )
        label = self.gmm(X)

        seq_output = self.forward(
            item_seq,
            label,
            interaction["collection_id"],
            interaction["bundle_value"],
            interaction["bundle_count"],
        )
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1) / self.temperature
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]

        # X = torch.cat([
        #     torch.mean(interaction["bundle_value_list"], dim=1, keepdim=True),
        #     torch.mean(interaction["bundle_count_list"], dim=1, keepdim=True),
        # ], dim=1).cpu().numpy()
        # self.gmm.fit(X)
        # label = self.gmm.predict(X)
        # label = torch.tensor(label).to(torch.long).to(config["device"])

        X = torch.stack(
            [
                interaction["bundle_value_list"],
                interaction["bundle_count_list"],
            ],
            dim=2,
        )
        label = self.gmm(X)

        seq_output = self.forward(
            item_seq,
            label,
            interaction["collection_id"],
            interaction["bundle_value"],
            interaction["bundle_count"],
        )
        test_item_emb = self.item_embedding.weight
        # no dropout for evaluation
        test_item_emb = F.normalize(test_item_emb, dim=-1)
        scores = (
            torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        )
        return scores


config["model"] = "RecNFT1202"
# init_seed(config["seed"], config["reproducibility"])
# logger initialization
init_logger(config)
logger.info(config)

# gmm = GaussianMixture(2, config["gmm_feat_size"] or 2)
# gmm.fit(dataset.inter_feat["bundle_value"])
# clustering_result = gmm.predict(dataset.inter_feat["bundle_value"].numpy(), probs=True)
# print(clustering_result)

config.model = RecNFT  # reset model for better debug
# model loading and initialization
model = RecNFT(
    config, train_data._dataset, n_collections=dataset.num("collection_id")
).to(config["device"])

logger.info(model)

# trainer loading and initialization
trainer = Trainer(config, model)

# model training
best_valid_score, best_valid_result = trainer.fit(
    train_data, valid_data, saved=True, show_progress=True
)

# model evaluation
test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)

logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
logger.info(set_color("test result", "yellow") + f": {test_result}")

logger.warning(
    {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }
)
