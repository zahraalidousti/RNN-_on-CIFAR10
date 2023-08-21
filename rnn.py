import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader

train_ds=dataset.CIFAR10(root='/root', train=True,transform=transforms.ToTensor(),download=True)
test_ds=dataset.CIFAR10(root='/root', train=False,transform=transforms.ToTensor(),download=True)


batch_size=32
train_dl=DataLoader(dataset=train_ds,batch_size=batch_size,shuffle=True,num_workers=2)
test_dl=DataLoader(dataset=test_ds,batch_size=batch_size,shuffle=True,num_workers=2)


image,target=next(iter(train_dl))
plt.figure(figsize=(5,5))
for i in range(18):
  plt.subplot(3,6,i+1)  
  img=torch.transpose(image[i],0,1)
  img=torch.transpose(img,1,2)
  plt.imshow(img)
  plt.axis('off')
plt.show() 

#RNN input ==>  batch_size,sequence , feature

device='cuda' if torch.cuda.is_available() else 'cpu'
device

#model
class RNN(nn.Module): 
   def __init__(self,input_feature,hidden_size=256,num_layer=2):   
       super(RNN,self),__init__()
       self.hidden_size=hidden_size
       self.num_layer=num_layer
       self.rnn=nn.RNN(input_feature,hidden_size,num_layer,batch_first=True)
       self.fc1=nn.Linear(in_features=hidden_size,out_features=num_class)   
      
   def forward(self,x):  
       x=x.permutr(0,2,3,1).flatten(2)
       h0=torch.randn(self.num_layer, x.size(0),self.hidden_size).to(device)
       out,_=self.rnn(x,h0)
       out=self.fc1(out[:,-1,:])
       return out
#end define model

model=RNN(input_feature=96).to(device)

citeration=nn.CrossEntropyLoss()

optimizer=optin.Adam(params=model.parameters(),lr=0.001)

epoch=6
#train
model.train()
for i in range(epoch):
      sumLoss=0
      for idx,(image,target) in enumerate(train_dl):

            image=image.to(device)
            target=target.to(device)

            optimizer.zero_grad()

            score=model(image)
            loss=citeration(score,target)

            sumLoss+=loss
            loss.backward()
            optimizer.step()

      print(f' in epoch number {i+1} is equal to { sumLoss }'  )

#check accuracy
def check_accuracy(dataloader,model):
      if dataloader.dataset.train:
           print('accuracy on train data is calculating...')
      else:
           print('accuracy on test data is calculating...')

      num_correct=0
      all_sample=0
      
      model.eval()
      with torch.no_grad():
             for x,y in dataloader:
                   x=x.to(device)
                   y=y.to(device)

                   score=model(x)
                   _,pred=score.max(1)

                   num_correct+=(pred==y).sum()
                   all_sample+=len(y)
      print(f'accuracy is { num_correct/all_sample }' )       

check_accuracy(train_dl,model)
check_accuracy(test_dl,model)
