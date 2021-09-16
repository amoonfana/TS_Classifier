# -*- coding: utf-8 -*-
import torch
import time

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, patience, device, save_path, log_interval):
    best_val_acc = 0
    time_train_total = 0
    time_val_avg = 0
    es_cnt = 0
    
    for epoch in range(0, n_epochs):
        # Train stage
        time_train = time.perf_counter()
        train_loss, train_acc = train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval)
        time_train_total += (time.perf_counter() - time_train)
        if epoch % log_interval == 0:
            print('Epoch: {}/{}. Train loss: {:.6f}\t Accuracy: {:.6f}'.format(epoch + 1, n_epochs, train_loss, train_acc))
        
        scheduler.step()
        
        # Validation stage
        time_val = time.perf_counter()
        val_loss, val_acc = val_epoch(val_loader, model, loss_fn, device)
        time_val_avg += (time.perf_counter() - time_val)
        if epoch % log_interval == 0:
            print('Epoch: {}/{}. Valid loss: {:.6f}\t Accuracy: {:.6f}'.format(epoch + 1, n_epochs, val_loss, val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # torch.save(model, save_path)
        
        # # Early stopping
        # if val_acc > best_val_acc:
        #     es_cnt = 0
        #     best_val_acc = val_acc
        #     # torch.save(model, save_path)
        # else:
        #     es_cnt += 1
            
        #     if es_cnt > patience:
        #         break
            
    return best_val_acc, time_train_total, time_val_avg/n_epochs            
        
def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval):
    accum_loss = 0
    accum_acc = 0
    
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)

        #Input-->NN-->feature-->loss-->output
        feature = model(data)
        output, loss = loss_fn(feature, label)
        
        #Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Accumulated loss
        accum_loss += loss.item()
        #Accumulated accuracy
        _, pred = output.max(1)
        num_correct = (pred == label).sum().item()
        acc = float(num_correct) / data.shape[0]
        accum_acc += acc

        #Print loss and accuracy while training
        # if batch_idx % log_interval == 0:
        #     print('Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), accum_loss/(batch_idx+1), accum_acc/(batch_idx+1)))
        
    return accum_loss/len(train_loader), accum_acc/len(train_loader)

def val_epoch(val_loader, model, loss_fn, device):
    accum_loss = 0
    accum_acc = 0
    
    with torch.no_grad():
        model.eval()
        for batch_idx, (data, label) in enumerate(val_loader):
            data = data.to(device)
            label = label.to(device)

            #Input-->NN-->feature-->loss-->output
            feature = model(data)
            output, loss = loss_fn(feature, label)

            #Accumulated loss
            accum_loss += loss.item()
            #Accumulated accuracy
            _, pred = output.max(1)
            num_correct = (pred == label).sum().item()
            acc = float(num_correct) / data.shape[0]
            accum_acc += acc

    return accum_loss/len(val_loader), accum_acc/len(val_loader)
