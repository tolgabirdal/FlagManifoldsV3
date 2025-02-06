import torch

def evaluate(model_, val_loader, DEVICE):
    model_.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No gradients needed during validation
        for val_support_images, val_support_labels, val_query_images, val_query_labels, _ in val_loader:
            # Obtain validation predictions
            val_preds = model_(val_support_images.to(DEVICE), val_support_labels.to(DEVICE), val_query_images.to(DEVICE))
            
            # Count correct predictions
            correct += (val_preds.argmax(dim=2).reshape(-1) == val_query_labels.to(DEVICE)).sum().item()
            total += val_query_labels.size(0)
            

    # Calculate validation accuracy
    val_accuracy = correct / total
    return val_accuracy