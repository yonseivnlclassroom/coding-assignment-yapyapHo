import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import KimchiMLP, KimchiCNN

class KimchiClassifier:
    def __init__(self, model_type):
        self.model_type = model_type
        if model_type == "MLP":
            self.model = KimchiMLP(3*64*64)
        elif model_type == "CNN":
            self.model = KimchiCNN()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # returns average test accuracy, class wise test accuracy
    def train_and_evaluate(self, train_dataset, val_dataset, test_dataset, num_epochs, batch_size):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size*2, shuffle=True) # batch size tuning.
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size)
        if batch_size > 100:
            batch_size = 100
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size)
        
     
        train_losses = []
        val_losses = []
        test_losses = []
        avg_train_accuracy = []
        avg_val_accuracy = []
        avg_test_accuracy = []
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                train_loss += loss.item()
                                
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == labels).sum().item()
                
                loss.backward()
                self.optimizer.step()

            train_losses.append(train_loss / len(train_loader))
            avg_train_accuracy.append(100*train_correct/len(train_dataset))

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == labels).sum().item()

            val_losses.append(val_loss / len(val_loader))
            avg_val_accuracy.append(100*val_correct/len(val_dataset))
            
            # Test
            self.model.eval()
            test_loss = 0.0
            class_correct = [0] * 11
            class_total = [0] * 11
            test_correct = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    test_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    test_correct += (predicted == labels).sum().item()

                    for label, prediction in zip(labels, predicted):
                        class_total[label.item()] += 1
                        if label == prediction:
                            class_correct[label.item()] += 1
                                           
            test_losses.append(test_loss / len(test_loader))
            avg_test_accuracy.append(100*test_correct/len(test_dataset))
            print(f"Epoch {epoch+1}/{num_epochs}, | Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}, Test Loss: {test_losses[-1]} ||| ", end='')
            print(f"Train Accuracy: {avg_train_accuracy[-1]}, Val Accuracy: {avg_val_accuracy[-1]}, Test Accuracy: {avg_test_accuracy[-1]}")

        return train_losses, val_losses, test_losses, class_correct, class_total, \
               avg_train_accuracy, avg_val_accuracy, avg_test_accuracy


    def plot_results(self, num_epochs, class_correct, class_total, avg_train_accuracy, avg_val_accuracy, avg_test_accuracy):
        epochs = range(1, num_epochs+1)
        
        # Plotting train, val, test accuracy
        plt.figure()
        plt.ylim([0, max(max(avg_train_accuracy), max(avg_val_accuracy), max(avg_test_accuracy))])
        plt.plot(epochs, avg_train_accuracy, label="Training Accuracy")
        plt.plot(epochs, avg_val_accuracy, label="Validation Accuracy")
        plt.plot(epochs, avg_test_accuracy, label="Test Accuracy")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training, Validation and Test Accuracy")
        plt.show()
        
        # Plotting class-wise test accuracy as a histogram
        plt.figure()
        class_accuracy = [correct / total for correct, total in zip(class_correct, class_total)]
        class_names = ['baechu', 'baik', 'boo', 'chong', 'got', 'kkak', 'moo', 'nabak', 'ohyee', 'pa', 'yeol']
        # used short names as original names are too long for plotting
        plt.bar(class_names, class_accuracy, color='skyblue')
        plt.xlabel("Class")
        plt.ylabel("Accuracy")
        plt.title("Class-wise Test Accuracy")
        plt.tight_layout()
        plt.show()
