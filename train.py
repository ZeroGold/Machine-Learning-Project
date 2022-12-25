import argparse
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from workspace_utils import active_session
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
parse = argparse.ArgumentParser(description="Train your network")
parse.add_argument("-data_dir", action="store", dest="data_dir", default="flowers")
parse.add_argument("-save_dir", action="store", dest="save_dir",
                  default = "./" )
parse.add_argument("-arch", action="store", dest="arch", default="alexnet", help ="Choose a network: densenet121, vgg13, or alexnet")
parse.add_argument("-learning_rate", action="store", dest="lr",default= 0.001)
parse.add_argument("-hidden_units", action="store", dest="hu", default=64)
parse.add_argument("-gpu", action="store", dest="gpu", default="cuda")
parse.add_argument("-epochs", action="store", dest="epochs", default = 20)
arg = parse.parse_args()

def main():
    data = arg.data_dir
    train_dir = data + "/train"
    test_dir = data + "/test"
    valid_dir = data + "/valid"
    model = None
    x = None
    
    data_transforms = transforms.Compose([transforms.RandomRotation(60),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    val_transforms = transforms.Compose([transforms.RandomRotation(60),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    image_datasets = datasets.ImageFolder(train_dir,transform=data_transforms)
    val_datasets = datasets.ImageFolder(valid_dir,transform=val_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
   


    dataloaders = torch.utils.data.DataLoader(image_datasets,batch_size=64,shuffle=True)
    val_dataloaders = torch.utils.data.DataLoader(val_datasets,batch_size=64,shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)
    

   
    if arg.arch ==  "densenet121":
        model = models.densenet121(pretrained=True)
        x = 1024
       

      
        
        
        
    if arg.arch ==  "vgg13":
        model = models.vgg13(pretrained=True)
        x = model.classifier[0].in_features
        
        
            
       
    if arg.arch ==  "alexnet":
        model = models.alexnet(pretrained=True)
        x = 9216
       
     
    
            
    
    for param in model.parameters():
        param.requires_grad = False 
        
    model.classifier = nn.Sequential(nn.Linear(x, arg.hu),
                                 nn.ReLU(),
                                 nn.Dropout(0.02),
                                 nn.Linear(arg.hu, 102),
                                 nn.LogSoftmax(dim=1))
        

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=arg.lr)
    
    
    
    de = arg.gpu
    if torch.cuda.is_available() is False:
        de = "cpu"
        print("Unable to use gpu")
        
    device = torch.device(de)
    epochs = int(arg.epochs)

    model.to(device)

    with active_session():
        for e in range(epochs):
            loss1 = 0
            for images, labels  in dataloaders:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                loss1+= loss.item()
            else:
                loss2 = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images,labels in val_dataloaders:
                            images, labels = images.to(device), labels.to(device)
                            logps = model(images)
                            loss = criterion(logps,labels)
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            eq = top_class == labels.view(*top_class.shape)
                            accuracy = torch.mean(eq.type(torch.FloatTensor))
                            print(f'Accuracy: {accuracy.item()*100}%')
                            loss2 += loss.item()
                model.train()
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                     "Training Loss: {:.3f}".format(loss1/len(dataloaders)),
                     "Validation Loss: {:.3f}".format(loss2/len(val_dataloaders)))
            
            
    model.class_to_idx = image_datasets.class_to_idx
    checkpoint = {'input_size':x,
              'hidden_layer_size': arg.hu,
              'output_size': 102,
              'arch': arg.arch,
              'learning_rate': arg.lr,
              'batch_size': 64,
              'classifier' : model.classifier,
              'epochs': epochs,
              'criterion': criterion,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}
    
    print("Model trained. Saving...")
    torch.save(checkpoint, arg.save_dir+'checkpoint.pth')
     

                
if __name__ == "__main__":
    main()
    

    