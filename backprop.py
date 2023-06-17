import torch
from tqdm import tqdm
from matplotlib import pyplot as plt

from dataset import CIFAR10


def get_correct_count(prediction, labels):
    return prediction.argmax(dim=1).eq(labels).sum().item()


def get_incorrect_preds(prediction, labels):
    prediction = prediction.argmax(dim=1)
    indices = prediction.ne(labels).nonzero().reshape(-1).tolist()
    return indices, prediction[indices].tolist(), labels[indices].tolist()


class Train(object):
    def __init__(self, model, device, criterion, train_loader, optimizer, l1=0):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.l1 = l1

        self.train_losses = list()
        self.train_acc = list()

    def run(self):
        self.model.train()
        pbar = tqdm(self.train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Predict
            pred = self.model(data)

            # Calculate loss
            loss = self.criterion(pred, target)
            if self.l1 > 0:
                loss += self.l1 * sum(p.abs().sum() for p in self.model.parameters())

            train_loss += loss.item() * len(data)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            correct += get_correct_count(pred, target)
            processed += len(data)

            pbar.set_description(
                desc=f"Train: Batch_id={batch_idx}, Average Loss={train_loss / processed:0.4f}, "
                     f"Accuracy={100 * correct / processed:0.2f}"
            )

        self.train_acc.append(100 * correct / processed)
        self.train_losses.append(train_loss / processed)

    def plot_stats(self):
        fig, axs = plt.subplots(1, 2, figsize=(8, 10))
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[0, 1].plot(self.train_acc)
        axs[0, 1].set_title("Training Accuracy")


class Test(object):
    def __init__(self, model, device, criterion, test_loader):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.test_loader = test_loader

        self.test_losses = list()
        self.test_acc = list()
        self.test_incorrect_pred = {
            "images": list(),
            "ground_truths": list(),
            "predicted_vals": list()
        }

    def run(self):
        self.model.eval()

        test_loss = 0
        correct = 0
        processed = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                pred = self.model(data)

                test_loss += self.criterion(pred, target, reduction="sum").item()

                correct += get_correct_count(pred, target)
                processed += len(data)

                ind, pred, truth = get_incorrect_preds(pred, target)
                self.test_incorrect_pred["images"] += data[ind]
                self.test_incorrect_pred["ground_truths"] += truth
                self.test_incorrect_pred["predicted_vals"] += pred

        test_loss /= processed
        self.test_acc.append(100 * correct / processed)
        self.test_losses.append(test_loss)

        print(f"Test: Average loss: {test_loss:0.4f}, Accuracy: {100 * correct / processed:0.2f}")

        return test_loss

    def show_incorrect(self):
        _ = plt.figure()
        for i in range(10):
            plt.subplot(5, 2, i + 1)
            plt.tight_layout()
            plt.imshow(CIFAR10.denormalise(self.test_incorrect_pred["images"][i].cpu()).permute(1, 2, 0))
            pred = self.test_incorrect_pred["predicted_vals"][i]
            truth = self.test_incorrect_pred["ground_truths"][i]
            if CIFAR10.classes is not None:
                pred = str(pred) + ':' + CIFAR10.classes[pred]
                truth = str(truth) + ':' + CIFAR10.classes[truth]
            plt.title(pred + "/" + truth)
            plt.xticks([])
            plt.yticks([])

    def plot_stats(self):
        fig, axs = plt.subplots(1, 2, figsize=(8, 10))
        axs[0, 0].plot(self.test_losses)
        axs[0, 0].set_title("Test Loss")
        axs[0, 1].plot(self.test_acc)
        axs[0, 1].set_title("Test Accuracy")
