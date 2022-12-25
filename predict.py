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




parse = argparse.ArgumentParser(description="Predict with your network")
parse.add_argument("--path",action="store", default="./flowers/test/1/image_06743.jpg", dest="path")
parse.add_argument("--top_k",action="store", default="5", dest="top_k")

parse.add_argument("--category_names", action="store", default="cat_to_name.json", dest="json")
parse.add_argument("--gpu", action="store",default="cpu", dest="gpu")


arg =  parse.parse_args()
def main():
    model = None
    file = "checkpoint.pth"
    checkpoint = torch.load(file)
    
    with open(arg.json,"r") as f:
        cat_to_name = json.load(f)
        
    if checkpoint["arch"] == "densenet121":
        model = models.densenet121(pretrained=True)
    if checkpoint["arch"] == "vgg13":
        model = models.vgg13(pretrained=True)
    if checkpoint["arch"] == "alexnet":
        model = models.alexnet(pretrained=True)
        
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    for param in model.parameters():
        param.requires_grad = False
        
   
    
    
    image = Image.open(arg.path)
    image = image.resize((256,256))
    image = image.crop((0,0,224,224))
    image = np.array(image)/255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image - mean) / std
   
    image = image.transpose((2, 0, 1))
    
    
    model.to(arg.gpu)
    model.eval()
    with torch.no_grad():
        image = torch.from_numpy(np.array([image])).float()
        image.to(arg.gpu)
        logps = model(image)
        ps = torch.exp(logps)
        topk = int(arg.top_k)
        p, classes = ps.topk(topk, dim=1)

        top_p = p.tolist()[0]
        top_classes = classes.tolist()[0]
        idx_to_class = {v:k for k, v in model.class_to_idx.items()}
        
        
        labels = []
        for c in top_classes:
            labels.append(cat_to_name[idx_to_class[c]])
        print("\u0332".join("\n Top {} Probabilities \n".format(arg.top_k)))
        for i in range(len(labels)):
            print("Class: {} \n   Probability: {:.2f}%".format(labels[i].upper(),top_p[i]*100))
if __name__ == "__main__":
    main()
    