from KimchiClassification import KimchiClassifier
from Dataset import KimchiDataset
from torchvision import transforms
from torch.utils.data import ConcatDataset


if __name__ == "__main__":
    # Define dataset paths
    train_data_dir = "dataset/train"
    val_data_dir = "dataset/val"
    test_data_dir = "dataset/test"
    
    transform_augument = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
    ])

    # Load and preprocess the dataset. Also contains augumented ones.
    train_dataset = KimchiDataset(train_data_dir)
    aug_train_dataset = KimchiDataset(train_data_dir, transform = transform_augument)
    val_dataset = KimchiDataset(val_data_dir)
    test_dataset = KimchiDataset(test_data_dir)
    
    # merge train_dataset and aug_train_dataset.
    train_dataset = ConcatDataset([train_dataset, aug_train_dataset])

    # num_epochs = 80
    # batch_size = 32
    
    # # Create and train MLP models
    # classifier = KimchiClassifier(model_type="MLP")

    # # Evaluate and visualize results
    # train_losses, val_losses, test_losses, class_correct, class_total, avg_train_accuracy, avg_val_accuracy, avg_test_accuracy \
    #     = classifier.train_and_evaluate(train_dataset, val_dataset, test_dataset, num_epochs, batch_size)

    # classifier.plot_results(num_epochs, class_correct, class_total, avg_train_accuracy, avg_val_accuracy, avg_test_accuracy)
    
    # Create and train CNN models
    num_epochs = 70
    batch_size = 64
    
    classifier = KimchiClassifier(model_type="CNN")

    train_losses, val_losses, test_losses, class_correct, class_total, avg_train_accuracy, avg_val_accuracy, avg_test_accuracy \
        = classifier.train_and_evaluate(train_dataset, val_dataset, test_dataset, num_epochs, batch_size)

    classifier.plot_results(num_epochs, class_correct, class_total, avg_train_accuracy, avg_val_accuracy, avg_test_accuracy)