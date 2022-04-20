import torch
from torch import nn


class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y: list, base_index: int) -> None:
        self.x = x
        self.y = y
        self.base_index = base_index

    def __getitem__(self, index):
        # Need to return x, y, g, index
        group = 0  # Assign a random group to these samples -- maybe group 0 which is the majority group
        dataset_index = index + self.base_index
        return self.x[index], self.y[index], group, dataset_index

    def __len__(self):
        return self.x.size(0)


class IdxDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, dataset_probe_identity):
        self.dataset = dataset
        self.dataset_probe_identity = dataset_probe_identity
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx], idx


class CustomConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, dataset, dataset_probe_identity):
        super().__init__([dataset, dataset_probe_identity])
        
        self.dataset = dataset
        self.dataset_probe_identity = dataset_probe_identity
        
        # Import the dataset properties
        self.n_classes = self.dataset.n_classes
        self.n_groups = self.dataset.n_groups
        self.group_str = self.dataset.group_str
        self.group_counts = self.dataset.group_counts


class CustomDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, dataset_probe_identity):
        super().__init__()
        
        self.dataset = dataset
        self.dataset_probe_identity = dataset_probe_identity
        
        # Import the dataset properties
        self.n_classes = self.dataset.n_classes
        self.n_groups = self.dataset.n_groups
        self.group_str = self.dataset.group_str
        self.group_counts = self.dataset.group_counts
    
    def __len__(self):
        return len(self.dataset_probe_identity)

    def __getitem__(self, idx):
        return self.dataset_probe_identity[idx]


def test_tensor(model, data, target, msg=None):
    assert torch.is_tensor(data) and torch.is_tensor(target)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    model.eval()
    with torch.no_grad():
        output = model(data)
        loss_vals = criterion(output, target)
        test_loss = float(loss_vals.mean())
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        total = len(data)
    
    test_acc = 100. * correct / total
    output_dict = dict(loss=test_loss, acc=test_acc, correct=correct, total=total, 
                       loss_vals=loss_vals.detach().cpu().numpy())
    
    if msg is not None:
        print(f"{msg} | Average loss: {test_loss:.4f} | Accuracy: {correct}/{total} ({test_acc:.2f}%)")
    
    return output_dict
