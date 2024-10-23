import torch


def test(model, device, test_loader, criterion, classes, test_losses, test_accs, misclassified_ims, correct_ims, is_last_epoch):
    model.eval()
    correct, test_loss = 0, 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            is_correct = pred.eq(target.view_ad(pred))

            if is_last_epoch:
                misclassified_inds = (is_correct==0).nonzero()[:, 0]
                for mis_ind in misclassified_inds:
                    if len(misclassified_ims)==25:
                        break
                    misclassified_ims.append({
                        "target": target[mis_ind].cpu().numpy(),
                        "pred": pred[mis_ind][0].cpu().numpy(),
                        "img": data[mis_ind].squeeze().cpu().numpy()
                    })

                correct_inds = (is_correct==1).nonzero()[:, 0]
                for ind in correct_inds:
                    if len(correct_ims)==25:
                        break
                    correct_ims.append({
                        "target": target[ind].cpu().numpy(),
                        "pred": pred[ind][0].cpu().numpy(),
                        "img": data[ind]
                    })
            correct += is_correct.sum().item()
    
    test_loss /= len(test_loader)
    test_losses.append(test_loss)

    test_acc = 100. * correct / len(test_loader.dataset)
    test_accs.append(test_acc)

    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc:.2f}%)\n")