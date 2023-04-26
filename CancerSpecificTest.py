import numpy as np
import pandas as pd
from numpy import *
from sklearn import metrics

from Utils.HGDataset import HGDataset
from Utils.utils import *

os.chdir("/private/xiongshuwen/CancerGene/workingspace/MyDriverGenes")


class Trainer:
    def __init__(self, model_name, epochs):
        super(Trainer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.hidden_channels = 64
        self.out_channels = 1
        self.num_heads = 4
        self.num_layers = 3
        self.lr = 0.001
        self.weight_decay = 5e-4
        self.model = None
        self.model_name = model_name
        self.optimizer = None
        self.criterion = None
        self.epochs = epochs
        self.repeats = 1
        self.folds = 5

    def inference(self, data, mask, fold):
        path = os.path.join("./SavedModels", self.model_name, "fold_{}.pth".format(fold))
        self.model = torch.load(f=path, map_location='cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        out = self.model(data.x_dict, data.edge_index_dict)
        pred = np.round(
            torch.sigmoid(out[data['gene'].label_index[mask]]).cpu().detach().numpy())
        labels = data['gene'].y[mask].cpu()
        ACC, F1, AUROC, AUPR = self.measure(pred, labels)
        return pred, ACC, F1, AUROC, AUPR

    def measure(self, pred, labels):
        ACC = metrics.accuracy_score(y_true=labels, y_pred=pred)
        F1 = metrics.f1_score(y_true=labels, y_pred=pred)
        AUROC = metrics.roc_auc_score(y_true=labels, y_score=pred)
        precision, recall, _ = metrics.precision_recall_curve(labels, pred)
        AUPR = metrics.auc(recall, precision)
        return ACC, F1, AUROC, AUPR

    def save_evaluation_indicators(self, indicators, cancer):
        path = os.path.join("SavedIndicators")

        if not os.path.exists(path):
            os.makedirs(path)
        #     写入评价指标
        file_name = os.path.join(path, "{}_CancerSpecific.xlsx".format(self.model_name))
        file = open(file_name, "a")

        file.write(cancer + " " + str(np.round(indicators[0], 4)) + " " + str(np.round(indicators[1], 4)) + " " +
                   str(np.round(indicators[2], 4)) + " " + str(np.round(indicators[3], 4)) + "\n")

        file.close()

    def run(self, data):
        data = data.to(self.device)

        cancers = ['KIRC', 'BRCA', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC',
                   'COAD', 'LUSC', 'CESC', 'KIRP']
        for i, cancer in enumerate(cancers):
            temp = data.clone().detach()
            ACC = []
            F1 = []
            AUROC = []
            AUPR = []
            temp['gene'].x[:, 0:i * 3] = 0
            temp['gene'].x[:, (i + 1) * 3:] = 0
            for fold in range(self.folds):
                test_pred, test_ACC, test_F1, test_AUROC, test_AUPR = self.inference(data=temp, mask=np.arange(data['gene'].y.shape[0]), fold=fold)
                ACC.append(test_ACC)
                F1.append(test_F1)
                AUROC.append(test_AUROC)
                AUPR.append(test_AUPR)
            mean_ACC = np.round(mean(ACC), 4)
            mean_F1 = np.round(mean(F1), 4)
            mean_AUROC = np.round(mean(AUROC), 4)
            mean_AUPR = np.round(mean(AUPR), 4)
            # Save the indicators
            indicators = [mean_ACC, mean_F1, mean_AUROC, mean_AUPR]
            self.save_evaluation_indicators(indicators, cancer)


if __name__ == '__main__':
    dataset = HGDataset(root="/private/xiongshuwen/CancerGene/workingspace/MyDriverGenes")
    data = dataset[0]

    model_name = "BEST"
    trainer = Trainer(model_name=model_name, epochs=1000)
    trainer.run(data=data)
