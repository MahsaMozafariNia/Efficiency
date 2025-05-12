import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms
from sklearn.utils import shuffle
import copy

import requests, zipfile, io

import math
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random









def get_splitCIFAR(seed=0, pc_valid=0.10, task_num = 0, split="", preprocess="Normalized", attack=None, modifier=None):

    if modifier == "OnlyCIFAR100":
        task_num += 1
        print("Offsetting tasknum to: ", task_num)
        
    if os.path.isfile(("./data/split_cifar/" + str(task_num) + "/x_train.bin")) == False:
        print("No dataset detected for similarity subsets. Creating new set prior to loading task.")
        make_splitcifar(seed=seed, pc_valid=pc_valid)
    
    if os.path.isfile(("./data/split_cifar/" + str(task_num) + "/x_extra_loader.bin")) == False:
        print("No dataset detected for similarity subsets. Creating new set prior to loading task.")
        make_splitcifar(seed=seed, pc_valid=pc_valid)


    data={}
    if attack and attack in ['gaussian_noise', 'gaussian_blur', 'saturate', 'rotate'] and split == "test":
        data['x']=torch.load(os.path.join(os.path.expanduser(('./data/split_cifar/' + str(task_num))), ('x_' + attack + '_test.bin')))
    else:
        data['x']=torch.load(os.path.join(os.path.expanduser(('./data/split_cifar/' + str(task_num))), ('x_'+split+'.bin')))
    data['y']=torch.load(os.path.join(os.path.expanduser(('./data/split_cifar/' + str(task_num))), ('y_'+split+'.bin')))



    if preprocess=="Unnormalized" and attack not in ['gaussian_blur', 'gaussian_noise', 'saturate', 'rotate']:
        if torch.min(data['x']) < -0.001:
            print("Data normalized, rescaling to 0:1 to match perturbation scale")
            
            if task_num == 0:
                ### CIFAR10
                mean= torch.tensor([0.4913725490196078, 0.4823529411764706, 0.4466666666666667]).view(1, 3, 1, 1)
                std = torch.tensor([0.24705882352941178, 0.24352941176470588, 0.2615686274509804]).view(1, 3, 1, 1)
            else: 
                ### CIFAR100
                mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
                std  = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)
            data['x'] = data['x'] * std + mean

    print("Data min: ", torch.min(data['x']), " new max: ", torch.max(data['x']))




    return data
    





def get_mixedCIFAR_KEFMNIST(seed = 0, pc_valid=0.1, task_num=0, split="", offset=False, preprocess="Normalized", attack=None, reduced=True, modifier=None):
    """
    Sequence: 
        0:CIFAR100 split, 
        1:Fashion MNIST, 
        2:CIFAR100 split, 
        3:EMNIST-Balanced, 
        4:CIFAR100 split, 
        5:KMNIST-49, 
    """
    if os.path.isfile(("./data/split_cifar/" + str(task_num) + "/x_train.bin")) == False:
        print("No dataset detected for cifar subsets. Creating new set prior to loading task.")
        make_splitcifar(seed=seed, pc_valid=pc_valid)
    
    data={}

    print("Loading task number: ", task_num, " for split: ", split, flush=True)

    if task_num == 1:
        data['x'] = torch.load(os.path.join(os.path.expanduser(('./data/FashionMNIST/')), ('x_'+split+'.bin')))
        data['y'] = torch.load(os.path.join(os.path.expanduser(('./data/FashionMNIST/')), ('y_'+split+'.bin')))
    elif task_num == 3:
        print("Loading EMNISTL", flush=True)
        data['x'] = torch.load(os.path.join(os.path.expanduser(('./data/EMNISTL/')), ('x_'+split+'.pt')))
        data['y'] = torch.load(os.path.join(os.path.expanduser(('./data/EMNISTL/')), ('y_'+split+'.pt')))
    elif task_num == 5:
        print("Loading KMNIST", flush=True)
        data['x'] = torch.load(os.path.join(os.path.expanduser(('./data/KMNIST10/')), ('x_'+split+'.pt')))
        data['y'] = torch.load(os.path.join(os.path.expanduser(('./data/KMNIST10/')), ('y_'+split+'.pt')))
    else:
        if task_num == 0:
            task_num = 1
        if attack and attack in ['gaussian_noise', 'gaussian_blur', 'saturate', 'rotate'] and split == "test":
            data['x']=torch.load(os.path.join(os.path.expanduser(('./data/split_cifar/' + str(task_num))), ('x_' + attack + '_test.bin')))
            print("Loading: ", 'x_' + attack + '_test.bin')
        else:
            data['x']=torch.load(os.path.join(os.path.expanduser(('./data/split_cifar/' + str(task_num))), ('x_'+split+'.bin')))
        data['y']=torch.load(os.path.join(os.path.expanduser(('./data/split_cifar/' + str(task_num))), ('y_'+split+'.bin')))


    if reduced:
        indices = torch.zeros(data['y'].size()).eq(1)

        quota = 1000
        quotas = {}
        for i, y in enumerate(data['y']):
          if y.item() not in quotas.keys():
            quotas[y.item()] = 0
          if quotas[y.item()] < quota:
            quotas[y.item()] += 1
            indices[i] = True


        data['x'] = data['x'][indices]
        data['y'] = data['y'][indices]




    if preprocess=="Unnormalized" and attack not in ['gaussian_blur', 'gaussian_noise', 'saturate', 'rotate']:
        if torch.min(data['x']) < -0.001:
            print("Data normalized, rescaling to 0:1 to match perturbation scale")
            
            if task_num == 1:
                ### FashionMNIST
                mean = torch.tensor([0.2860, 0.2860, 0.2860]).view(1, 3, 1, 1)
                std  = torch.tensor([0.3530, 0.3530, 0.3530]).view(1, 3, 1, 1)
            elif task_num == 3:
                ### EMNIST
                print("Denormalizing EMNISTL")
                mean = torch.tensor([0.1724, 0.1724, 0.1724]).view(1, 3, 1, 1)
                std  = torch.tensor([0.3084, 0.3084, 0.3084]).view(1, 3, 1, 1)
                # print("Denormalizing EMNIST")
                # mean = torch.tensor([0.1735, 0.1735, 0.1735]).view(1, 3, 1, 1)
                # std  = torch.tensor([0.3099, 0.3099, 0.3099]).view(1, 3, 1, 1)
            elif task_num == 5:
                ### KMNIST
                mean = torch.tensor([0.1915,0.1915,0.1915]).view(1, 3, 1, 1)
                std  = torch.tensor([0.3149,0.3149, 0.3149]).view(1, 3, 1, 1)
            else: 
                ### CIFAR100
                mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
                std  = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)
            data['x'] = data['x'] * std + mean

    print("Data min: ", torch.min(data['x']), " new max: ", torch.max(data['x']))


    return data
    




def get_Synthetic(task_num=0, split="", modifier="ai", preprocess="Normalized", attack=None):

    data={}

    if split == "valid":
        split = "val"

    taskDict = {0:"ADM", 1:"BigGAN", 2:"Midjourney", 3:"glide", 4:"stable_diffusion_v_1_4", 5:"VQDM"}
    generator = taskDict[task_num]

    loadPath = os.path.join(".", "data/Synthetic", generator, str(task_num), split, modifier) 
    print("Loading Synthetic dataset from: ", loadPath)

    if attack and attack in ['gaussian_noise', 'gaussian_blur', 'saturate', 'rotate'] and split == "test":
        data['x']=torch.load(os.path.join(loadPath, ('X_' + attack + '.pt')))
    else:
        data['x']=torch.load(os.path.join(loadPath, 'X.pt'))
    data['y']=torch.load(os.path.join(loadPath, 'y.pt'))


    numImages = data['y'].size(0)
    # print("Number of images: ", numImages)

    numSets = numImages // 10


    data['x'] = data['x'][:(numSets*10)]
    data['y'] = data['y'][:(numSets*10)]

    numImages = data['y'].size(0)
    print("Number of images: ", numImages)

    ### Map original labels to new sequential indices compatible with a new classifier
    original_labels = torch.unique(data['y']).tolist()  
    label_map = {original_label: new_label for new_label, original_label in enumerate(original_labels)}
    mapped_y = torch.tensor([label_map[label.item()] for label in data['y']])


    data['y'] = mapped_y

    # Unnormalize the data back into a [0,1] range for appropriately scaled adversarial attacks
    if preprocess=="Unnormalized" and attack not in ['gaussian_blur', 'gaussian_noise', 'saturate', 'rotate']:
        if torch.min(data['x']) < -0.001:
            
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

            data['x'] = data['x'] * std + mean

    print("Data min: ", torch.min(data['x']), " new max: ", torch.max(data['x']))


    return data
    
    



def get_Synthetic_SingleGenerator(task_num=0, split="", generator='ADM', modifier="ai", preprocess="Normalized", attack=None):

    data={}

    if split == "valid":
        split = "val"


    loadPath = os.path.join(".", "data/Synthetic", generator, str(task_num), split, modifier) 
    print("Loading Synthetic dataset from: ", loadPath)

    if attack and attack in ['gaussian_noise', 'gaussian_blur', 'saturate', 'rotate'] and split == "test":
        data['x']=torch.load(os.path.join(loadPath, ('X_' + attack + '.pt')))
    else:
        data['x']=torch.load(os.path.join(loadPath, 'X.pt'))
    data['y']=torch.load(os.path.join(loadPath, 'y.pt'))


    numImages = data['y'].size(0)
    # print("Number of images: ", numImages)

    numSets = numImages // 10


    data['x'] = data['x'][:(numSets*10)]
    data['y'] = data['y'][:(numSets*10)]

    numImages = data['y'].size(0)
    print("Number of images: ", numImages)

    ### Map original labels to new sequential indices compatible with a new classifier
    original_labels = torch.unique(data['y']).tolist()  
    label_map = {original_label: new_label for new_label, original_label in enumerate(original_labels)}
    mapped_y = torch.tensor([label_map[label.item()] for label in data['y']])

    data['y'] = mapped_y

    ### Unnormalize the data back into a [0,1] range for appropriately scaled adversarial attacks
    if preprocess=="Unnormalized" and attack not in ['gaussian_blur', 'gaussian_noise', 'saturate', 'rotate']:
        if torch.min(data['x']) < -0.001:
            
            ### Imagenet Means and Biases
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

            data['x'] = data['x'] * std + mean

    print("Data min: ", torch.min(data['x']), " new max: ", torch.max(data['x']))


    return data
    
    































### Functions for generating datasets, may be missing some subsequent processing steps

    
    

### Download and set up the split CIFAR-10/100 dataset
def make_splitcifar(seed=0, pc_valid=0.2):
    print("Making SplitCifar", flush=True)
   
    
    dat={}
    data={}
    taskcla=[]
    size=[3,32,32]
    
    
    # CIFAR10
    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]
    
    
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])

        
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    # CIFAR10
    dat['train']        = datasets.CIFAR10('./data/',train=True,download=True,transform=train_transform)
    dat['extra_loader'] = datasets.CIFAR10('./data/',train=True,download=True,transform=test_transform)
    dat['test']         = datasets.CIFAR10('./data/',train=False,download=True, transform=test_transform)
    
    print("train equals extra: ", dat['train'].targets == dat['extra_loader'].targets)
    print("train: ", dat['train'].targets[:10])
    print("extra: ", dat['extra_loader'].targets[:10])
    
    print("Loaded CIFAR10", flush=True)
    data['name']='cifar10'
    data['ncla']=10
    data['train']={'x': [],'y': []}
    data['extra_loader']={'x': [],'y': []}
    data['valid']={'x': [],'y': []}
    data['test']={'x': [],'y': []}
    
    ### Get a shuffled set of validation data without shuffling the train and extra datasets directly to ensure they match
    nvalid=int(0.1*50000)
    random_indices = torch.randperm(len(dat['train'].targets))
    random_indices_list = random_indices.tolist()
    ### Shuffle the train and extra loader in tandem using the same random index order
    dat['train'].data    = dat['train'].data[random_indices]
    dat['train'].targets = [dat['train'].targets[i] for i in random_indices_list]
    dat['extra_loader'].data    = dat['extra_loader'].data[random_indices]
    dat['extra_loader'].targets = [dat['extra_loader'].targets[i] for i in random_indices_list]

    # print("Train Extra Equality: ", np.all(dat['train'].data == dat['extra_loader'].data))
    print("shuffled train: ", dat['train'].targets[:10])
    print("shuffled extra: ", dat['extra_loader'].targets[:10])

    for s in ['train','test', 'extra_loader']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        if s == 'train':
            for n, (image,target) in enumerate(loader):
                if n < nvalid: 
                    data['valid']['x'].append(image)
                    data['valid']['y'].append(target.numpy()[0])
                else:
                    data['train']['x'].append(image)
                    data['train']['y'].append(target.numpy()[0])
                    
            data['train']['x']=torch.stack(data['train']['x']).view(-1,size[0],size[1],size[2])
            data['train']['y']=torch.LongTensor(np.array(data['train']['y'],dtype=int)).view(-1)

            data['valid']['x']=torch.stack(data['valid']['x']).view(-1,size[0],size[1],size[2])
            data['valid']['y']=torch.LongTensor(np.array(data['valid']['y'],dtype=int)).view(-1)
        
            os.makedirs(('./data/split_cifar/' + str(0)) ,exist_ok=True)
            torch.save(data['train']['x'], ('./data/split_cifar/'+ str(0) + '/x_' + 'train' + '.bin'))
            torch.save(data['train']['y'], ('./data/split_cifar/'+ str(0) + '/y_' + 'train' + '.bin'))
            torch.save(data['valid']['x'], ('./data/split_cifar/'+ str(0) + '/x_' + 'valid' + '.bin'))
            torch.save(data['valid']['y'], ('./data/split_cifar/'+ str(0) + '/y_' + 'valid' + '.bin'))

            data['train']={'x': [],'y': []}
            data['valid']={'x': [],'y': []}

        elif s == 'extra_loader':
            for n, (image,target) in enumerate(loader):
                if n >= nvalid: 
                    data['extra_loader']['x'].append(image)
                    data['extra_loader']['y'].append(target.numpy()[0])
            data['extra_loader']['x']=torch.stack(data['extra_loader']['x']).view(-1,size[0],size[1],size[2])
            data['extra_loader']['y']=torch.LongTensor(np.array(data['extra_loader']['y'],dtype=int)).view(-1)
            
            os.makedirs(('./data/split_cifar/' + str(0)) ,exist_ok=True)
            torch.save(data['extra_loader']['x'], ('./data/split_cifar/'+ str(0) + '/x_' + 'extra_loader' + '.bin'))
            torch.save(data['extra_loader']['y'], ('./data/split_cifar/'+ str(0) + '/y_' + 'extra_loader' + '.bin'))

            data['extra_loader']={'x': [],'y': []}

        else:
            for n, (image,target) in enumerate(loader):
                data['test']['x'].append(image)
                data['test']['y'].append(target.numpy()[0])

            data['test']['x']=torch.stack(data['test']['x']).view(-1,size[0],size[1],size[2])
            data['test']['y']=torch.LongTensor(np.array(data['test']['y'],dtype=int)).view(-1)    
            torch.save(data['test']['x'], ('./data/split_cifar/'+ str(0) + '/x_' + 'test' + '.bin'))
            torch.save(data['test']['y'], ('./data/split_cifar/'+ str(0) + '/y_' + 'test' + '.bin'))

            data['test']={'x': [],'y': []}
            
  
        
    
    
    
    
    
    
    
    
    
    
    
    print("Making Split Cifar100", flush=True)
    # CIFAR100
    dat={}
    
    
    mean = [0.5071, 0.4867, 0.4408]
    std  = [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
    
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    dat={}
    data={}
    
    dat['train']         = datasets.CIFAR100('./data/',train=True,  download=True, transform=train_transform)
    dat['extra_loader']  = datasets.CIFAR100('./data/',train=True,  download=True, transform=test_transform)
    dat['test']          = datasets.CIFAR100('./data/',train=False, download=True, transform=test_transform)
    
    print("Loaded CIFAR100", flush=True)
    
    ntasks = 5
    ncla = 20
    for n in range(ntasks):
        data[n]={}
        data[n]['name']='cifar100'
        data[n]['ncla']=20
        data[n]['train']={'x': [],'y': []}
        data[n]['valid']={'x': [],'y': []}
        data[n]['extra_loader']={'x': [],'y': []}
        data[n]['extra_valid'] ={'x': [],'y': []}
        data[n]['test']={'x': [],'y': []}
        # train_list= [[],[],[],[],[]]
        # extra_list = [[],[],[],[],[]]
    

    ### Get a shuffled set of validation data without shuffling the train and extra datasets directly to ensure they match
    random_indices = torch.randperm(len(dat['train'].targets))
    random_indices_list = random_indices.tolist()
    ### Shuffle the train and extra loader in tandem using the same random index order
    dat['train'].data    = dat['train'].data[random_indices]
    dat['train'].targets = [dat['train'].targets[i] for i in random_indices_list]
    dat['extra_loader'].data    = dat['extra_loader'].data[random_indices]
    dat['extra_loader'].targets = [dat['extra_loader'].targets[i] for i in random_indices_list]
    
    # print("Train Extra Equality: ", np.all(dat['train'].data == dat['extra_loader'].data))
    print("shuffled train: ", dat['train'].targets[:10])
    print("shuffled extra: ", dat['extra_loader'].targets[:10])

    ### Number of validation samples to be split off per task
    nvalid=int(0.1*(len(dat['train'].targets)/ntasks))


    for s in ['train','test', 'extra_loader']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        if s == "train":
            for n, (image,target) in enumerate(loader):
                task_idx = (target.numpy()[0] // ncla)
                
                ### If the matching task's validation data is not full, add it to valid, otherwise add it to train
                if len(data[task_idx]['valid']['y']) < nvalid:
                    # train_list[task_idx].append(n)
                    data[task_idx]['valid']['x'].append(image)
                    data[task_idx]['valid']['y'].append(target.numpy()[0] % ncla)
                else:
                    data[task_idx]['train']['x'].append(image)
                    data[task_idx]['train']['y'].append(target.numpy()[0] % ncla)
    
            # print('\n treain list', train_list[0])
            for t in range(ntasks):
                data[t]['train']['x']=torch.stack(data[t]['train']['x']).view(-1,size[0],size[1],size[2])
                data[t]['train']['y']=torch.LongTensor(np.array(data[t]['train']['y'],dtype=int)).view(-1)
                
                data[t]['valid']['x']=torch.stack(data[t]['valid']['x']).view(-1,size[0],size[1],size[2])
                data[t]['valid']['y']=torch.LongTensor(np.array(data[t]['valid']['y'],dtype=int)).view(-1)
                
                # print(data[t]['train']['x'].shape,flush=True)
                # print(data[t]['valid']['x'].shape,flush=True)
                os.makedirs(('./data/split_cifar/' + str(t+1)) ,exist_ok=True)
                torch.save(data[t]['train']['x'], ('./data/split_cifar/'+ str(t+1) + '/x_' + 'train' + '.bin'))
                torch.save(data[t]['train']['y'], ('./data/split_cifar/'+ str(t+1) + '/y_' + 'train' + '.bin'))
                torch.save(data[t]['valid']['x'], ('./data/split_cifar/'+ str(t+1) + '/x_' + 'valid' + '.bin'))
                torch.save(data[t]['valid']['y'], ('./data/split_cifar/'+ str(t+1) + '/y_' + 'valid' + '.bin'))

                data[t]['train']={'x': [],'y': []}
                data[t]['valid']={'x': [],'y': []}
            
            
            
        if s == 'extra_loader':
            for n, (image,target) in enumerate(loader):
                task_idx = (target.numpy()[0] // ncla)
    
                ### Splits validation data same as train, but stores it in a placeholder buffer which does not get saved
                #*# If we wanted to save memory we could simply track an integer of how many samples we've skipped over in each task until nvalid samples are skipped
                if len(data[task_idx]['extra_valid']['y']) < nvalid:
                    # train_list[task_idx].append(n)
                    data[task_idx]['extra_valid']['x'].append(image)
                    data[task_idx]['extra_valid']['y'].append(target.numpy()[0] % ncla)
                else:
                    data[task_idx]['extra_loader']['x'].append(image)
                    data[task_idx]['extra_loader']['y'].append(target.numpy()[0] % ncla)
                    
            for t in range(ntasks):
                data[t]['extra_loader']['x']=torch.stack(data[t]['extra_loader']['x']).view(-1,size[0],size[1],size[2])
                data[t]['extra_loader']['y']=torch.LongTensor(np.array(data[t]['extra_loader']['y'],dtype=int)).view(-1)

                os.makedirs(('./data/split_cifar/' + str(t+1)) ,exist_ok=True)
                torch.save(data[t]['extra_loader']['x'], ('./data/split_cifar/'+ str(t+1) + '/x_' + 'extra_loader' + '.bin'))
                torch.save(data[t]['extra_loader']['y'], ('./data/split_cifar/'+ str(t+1) + '/y_' + 'extra_loader' + '.bin'))
                
                data[t]['extra_loader']={'x': [],'y': []}
            
               


        if s == 'test':
            for image,target in loader:
                task_idx = (target.numpy()[0] // ncla)
                data[task_idx]['test']['x'].append(image)
                data[task_idx]['test']['y'].append(target.numpy()[0] % ncla)
        
            for t in range(ntasks):
                data[t]['test']['x']=torch.stack(data[t]['test']['x']).view(-1,size[0],size[1],size[2])
                data[t]['test']['y']=torch.LongTensor(np.array(data[t]['test']['y'],dtype=int)).view(-1)
                
                # print(data[t]['test']['x'].shape,flush=True)
                torch.save(data[t]['test']['x'], ('./data/split_cifar/'+ str(t+1) + '/x_' + 'test' + '.bin'))
                torch.save(data[t]['test']['y'], ('./data/split_cifar/'+ str(t+1) + '/y_' + 'test' + '.bin'))
                
                data[t]['test']={'x': [],'y': []}
                




    
    
def make_PMNIST(seed=0, pc_valid=0.1):
    
    mnist_train = datasets.MNIST('./data/', train = True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),
                                transforms.Resize(32)]), download = True)        
    mnist_extraloader = datasets.MNIST('./data/', train = True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),
                                transforms.Resize(32)]), download = True)  
    mnist_test = datasets.MNIST('./data/', train = False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),
                                transforms.Resize(32)]), download = True)        

    dat={}
    data={}
    taskcla=[]
    size=[3,32,32]    
    os.makedirs('./data/PMNIST', exist_ok =True)
    
    dat['train'] = mnist_train
    dat['extra_loader']  = mnist_extraloader
    dat['test'] = mnist_test
    
    ### Get a shuffled set of validation data without shuffling the train and extra datasets directly to ensure they match
    random_indices = torch.randperm(len(dat['train'].targets))
    random_indices_list = random_indices.tolist()

    ### Shuffle the train and extra loader in tandem using the same random index order
    dat['train'].data    = dat['train'].data[random_indices]
    dat['train'].targets = [dat['train'].targets[i] for i in random_indices_list]
    dat['extra_loader'].data    = dat['extra_loader'].data[random_indices]
    dat['extra_loader'].targets = [dat['extra_loader'].targets[i] for i in random_indices_list]
    
    nvalid=int(0.1*(len(dat['train'].targets)))
    # print("Train Extra Equality: ", np.all(dat['train'].data == dat['extra_loader'].data))
    print("shuffled train: ", dat['train'].targets[:10])
    print("shuffled extra: ", dat['extra_loader'].targets[:10])

    ### Prepare the data variable and lists of label indices for further processing
    for t in range(0,6):
        print("Making task: ", t, flush=True)
        data={}
        data['name']='PMNIST'
        data['ncla']=10
        data['train']={'x': [],'y': []}
        data['valid']={'x': [],'y': []}
        data['extra_loader']={'x': [],'y': []}
        data['test']={'x': [],'y': []}

        torch.manual_seed(t)
        taskperm = torch.randperm((32*32))
        # ### Extract only the appropriately labeled samples for each of the subsets
        for s in ['train','test', 'extra_loader']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            


        # for s in ['train','test', 'extra_loader']:
        #     loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
    
            if s == "train":
                for n, (image,target) in enumerate(loader):
                    # print("nvalid: ", nvalid, flush=True)
                    ### Flatten the (1,32,32) image into (1,1024)
                    image = torch.flatten(image)
                    image = image[taskperm]
                    image = image.view(1,32,32)
                    ### Should give shape (3,32,32)
                    image = torch.cat((image,image,image), dim=0)
    


                    ### If the matching task's validation data is not full, add it to valid, otherwise add it to train
                    if len(data['valid']['y']) < nvalid:
                        # train_list[task_idx].append(n)
                        data['valid']['x'].append(image)
                        data['valid']['y'].append(target.numpy()[0])
                    else:
                        data['train']['x'].append(image)
                        data['train']['y'].append(target.numpy()[0])
                        

                data['train']['x']=torch.stack(data['train']['x']).view(-1,size[0],size[1],size[2])
                data['train']['y']=torch.LongTensor(np.array(data['train']['y'],dtype=int)).view(-1)
                data['valid']['x']=torch.stack(data['valid']['x']).view(-1,size[0],size[1],size[2])
                data['valid']['y']=torch.LongTensor(np.array(data['valid']['y'],dtype=int)).view(-1)
                                    
                os.makedirs(('./data/PMNIST/' + str(t)) ,exist_ok=True)
                torch.save(data['train']['x'], ('./data/PMNIST/'+ str(t) + '/x_' + 'train' + '.bin'))
                torch.save(data['train']['y'], ('./data/PMNIST/'+ str(t) + '/y_' + 'train' + '.bin'))
                torch.save(data['valid']['x'], ('./data/PMNIST/'+ str(t) + '/x_' + 'valid' + '.bin'))
                torch.save(data['valid']['y'], ('./data/PMNIST/'+ str(t) + '/y_' + 'valid' + '.bin'))
    
                data['train']={'x': [],'y': []}
                data['valid']={'x': [],'y': []}
            
                
                
            if s == 'extra_loader':
                for n, (image,target) in enumerate(loader):

                    ### Flatten the (1,32,32) image into (1,1024)
                    image = torch.flatten(image)
                    image = image[taskperm]
                    image = image.view(1,32,32)
                    ### Should give shape (3,32,32)
                    image = torch.cat((image,image,image), dim=0)

                    if n >= nvalid: 
                        # print("Storing n: ", n, flush=True)
                        data['extra_loader']['x'].append(image)
                        data['extra_loader']['y'].append(target.numpy()[0])
                data['extra_loader']['x']=torch.stack(data['extra_loader']['x']).view(-1,size[0],size[1],size[2])
                data['extra_loader']['y']=torch.LongTensor(np.array(data['extra_loader']['y'],dtype=int)).view(-1)
            
                torch.save(data['extra_loader']['x'], ('./data/PMNIST/'+ str(t) + '/x_' + 'extra_loader' + '.bin'))
                torch.save(data['extra_loader']['y'], ('./data/PMNIST/'+ str(t) + '/y_' + 'extra_loader' + '.bin'))
                
                data['extra_loader']={'x': [],'y': []}
            
                   
    
    
            if s == 'test':
                # for image,target in loader:
                for n, (image,target) in enumerate(loader):
                    ### Flatten the (1,32,32) image into (1,1024)
                    image = torch.flatten(image)
                    image = image[taskperm]
                    image = image.view(1,32,32)
                    ### Should give shape (3,32,32)
                    image = torch.cat((image,image,image), dim=0)
    

                    data['test']['x'].append(image)
                    data['test']['y'].append(target.numpy()[0])
            
                data['test']['x']=torch.stack(data['test']['x']).view(-1,size[0],size[1],size[2])
                data['test']['y']=torch.LongTensor(np.array(data['test']['y'],dtype=int)).view(-1)
                
                torch.save(data['test']['x'], ('./data/PMNIST/'+ str(t) + '/x_' + 'test' + '.bin'))
                torch.save(data['test']['y'], ('./data/PMNIST/'+ str(t) + '/y_' + 'test' + '.bin'))
                    
                data['test']={'x': [],'y': []}
                
                            
            
            
    
    

### Adapted from https://github.com/rcamino/pytorch-notebooks/blob/master/Train%20Torchvision%20Models%20with%20Tiny%20ImageNet-200.ipynb
def make_TinyImagenet(seed=0, pc_valid=0.1):
    
    ### URL for Tiny Imagenet
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    
    ### Send HTTP request to the URL of the file
    r = requests.get(url)
    
    ### If successful, then unzip
    if r.status_code == requests.codes.ok:
        print('Download successful. Unzipping file.')
        
        ### Extract the content of the zip file
        z = zipfile.ZipFile(io.BytesIO(r.content))

        z.extractall()
        # z.extractall(path=TIpath)
    else:
        print('Failed to download the file.')
    
    directory = "./tiny-imagenet-200/"
    TIpath = "./data/Tiny Imagenet/"
    os.makedirs(TIpath ,exist_ok=True)

    num_classes = 200
    batch_size = 256
    
    
    #!# We will process and split the training data into training and validation sets
    
    # the magic normalization parameters come from the example
    transform_mean = np.array([ 0.485, 0.456, 0.406 ])
    transform_std = np.array([ 0.229, 0.224, 0.225 ])
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=64,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = transform_mean, std = transform_std),
    ])
    
    traindir = os.path.join(directory, "train")
    # be careful with this set, the labels are not defined using the directory structure
    
    train = datasets.ImageFolder(traindir, train_transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    
    assert num_classes == len(train_loader.dataset.classes)

    print("Processing labels",flush=True)
    
    small_labels = {}
    with open(os.path.join(directory, "words.txt"), "r") as dictionary_file:
        line = dictionary_file.readline()
        while line:
            label_id, label = line.strip().split("\t")
            small_labels[label_id] = label
            line = dictionary_file.readline()
            
    labels = {}
    label_ids = {}
    for label_index, label_id in enumerate(train_loader.dataset.classes):
        ### label_id is the string code, label is the english word(s), and label_index is the integer class number
        label = small_labels[label_id]
        labels[label_index] = label
        label_ids[label_id] = label_index
        
    ### Number of images to be split off for validation
    nvalid = math.floor(pc_valid * 100000)        
    train_images = []
    train_labels = []
    valid_images = []
    valid_labels = []
    print("nvalid: ", nvalid, flush=True)
    for index, (images, labels) in enumerate(train_loader):
        ### If the next batch can fit in the validation set then we store it there
        if (index * batch_size) + batch_size <= nvalid:
            valid_images.append(images)
            valid_labels.append(labels)
        ### If we're already over the number of validation samples needed then append all images to the training set
        elif (index * batch_size) > nvalid:
            train_images.append(images)
            train_labels.append(labels)
        else:
            ### Number of validation images remaining to be stored
            remaining_valid = int(nvalid-(index * batch_size))
            valid_images.append(images[:remaining_valid])
            valid_labels.append(labels[:remaining_valid])
            train_images.append(images[remaining_valid:])
            train_labels.append(labels[remaining_valid:])
            

    # Concatenate the lists of batches into a single tensor
    valid_images = torch.cat(valid_images, dim=0)
    valid_labels = torch.cat(valid_labels, dim=0)
    os.makedirs("./data/Tiny Imagenet/valid" ,exist_ok=True)
    torch.save(valid_images, "./data/Tiny Imagenet/valid/X.pt")
    torch.save(valid_labels, "./data/Tiny Imagenet/valid/y.pt")

    print("Size of split validation set: ", valid_images.shape)
    ### Overwrite images to free from memory before concatenating the training data
    valid_images = []
    valid_labels = []


    # Concatenate the lists of batches into a single tensor
    train_images = torch.cat(train_images, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    os.makedirs("./data/Tiny Imagenet/train" ,exist_ok=True)
    torch.save(train_images, "./data/Tiny Imagenet/train/X.pt")
    torch.save(train_labels, "./data/Tiny Imagenet/train/y.pt")
    
    print("Size of training set: ", train_images.shape)
    train_images = []
    train_labels = []


    
    
    #!# Repeat for the original validation set which we will use as the test data
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = transform_mean, std = transform_std),
    ])
    
    valdir = os.path.join(directory, "val")
    val = datasets.ImageFolder(valdir, val_transform)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)
    
    
    val_label_map = {}
    with open(os.path.join(directory, "val/val_annotations.txt"), "r") as val_label_file:
        line = val_label_file.readline()
        while line:
            file_name, label_id, _, _, _, _ = line.strip().split("\t")
            val_label_map[file_name] = label_id
            line = val_label_file.readline()
            
    for i in range(len(val_loader.dataset.imgs)):
        file_path = val_loader.dataset.imgs[i][0]
    
        file_name = os.path.basename(file_path)
        label_id = val_label_map[file_name]
    
        val_loader.dataset.imgs[i] = (file_path, label_ids[label_id])
        

    val_images = []
    val_labels = []
    
    for index, epoch in enumerate(val_loader):
        # print(index)
        images, labels = epoch
        val_images.append(images)
        val_labels.append(labels)
    
    # Concatenate the lists of batches into a single tensor
    val_images = torch.cat(val_images, dim=0)
    val_labels = torch.cat(val_labels, dim=0)
    
    
    
    os.makedirs("./data/Tiny Imagenet/test" ,exist_ok=True)
    torch.save(val_images, "./data/Tiny Imagenet/test/X.pt")
    torch.save(val_labels, "./data/Tiny Imagenet/test/y.pt")








    
def make_FashionMNIST(seed=0, pc_valid=0.1):
    
    mnist_train = datasets.FashionMNIST('./data/', train = True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.2860,), (0.3530,)),
                                transforms.Resize(32)]), download = False)        
    mnist_test = datasets.FashionMNIST('./data/', train = False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.2860,), (0.3530,)),
                                transforms.Resize(32)]), download = False)        

    dat={}
    data={}
    taskcla=[]
    size=[3,32,32]    
    
    dat['train'] = mnist_train
    dat['test'] = mnist_test
    
    ### Get a shuffled set of validation data without shuffling the train and extra datasets directly to ensure they match
    random_indices = torch.randperm(len(dat['train'].targets))
    random_indices_list = random_indices.tolist()

    ### Shuffle the train and extra loader in tandem using the same random index order
    dat['train'].data    = dat['train'].data[random_indices]
    dat['train'].targets = [dat['train'].targets[i] for i in random_indices_list]
    
    nvalid=int(0.1*(len(dat['train'].targets)))
    # print("Train Extra Equality: ", np.all(dat['train'].data == dat['extra_loader'].data))
    print("shuffled train: ", dat['train'].targets[:10])

    ### Prepare the data variable and lists of label indices for further processing
    data={}
    data['ncla']=10
    data['train']={'x': [],'y': []}
    data['valid']={'x': [],'y': []}
    data['test']={'x': [],'y': []}

    # ### Extract only the appropriately labeled samples for each of the subsets
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        
        if s == "train":
            for n, (image,target) in enumerate(loader):
                # print("nvalid: ", nvalid, flush=True)
                ### Flatten the (1,32,32) image into (1,1024)
                image = torch.flatten(image)
                image = image.view(1,32,32)
                ### Should give shape (3,32,32)
                image = torch.cat((image,image,image), dim=0)


                ### If the matching task's validation data is not full, add it to valid, otherwise add it to train
                if len(data['valid']['y']) < nvalid:
                    data['valid']['x'].append(image)
                    data['valid']['y'].append(target.numpy()[0])
                else:
                    data['train']['x'].append(image)
                    data['train']['y'].append(target.numpy()[0])
                    

            data['train']['x']=torch.stack(data['train']['x']).view(-1,size[0],size[1],size[2])
            data['train']['y']=torch.LongTensor(np.array(data['train']['y'],dtype=int)).view(-1)
            data['valid']['x']=torch.stack(data['valid']['x']).view(-1,size[0],size[1],size[2])
            data['valid']['y']=torch.LongTensor(np.array(data['valid']['y'],dtype=int)).view(-1)
                                
            torch.save(data['train']['x'], ('./data/FashionMNIST/x_' + 'train' + '.bin'))
            torch.save(data['train']['y'], ('./data/FashionMNIST/y_' + 'train' + '.bin'))

            torch.save(data['train']['x'], ('./data/FashionMNIST/x_' + 'extra_loader' + '.bin'))
            torch.save(data['train']['y'], ('./data/FashionMNIST/y_' + 'extra_loader' + '.bin'))
            
            torch.save(data['valid']['x'], ('./data/FashionMNIST/x_' + 'valid' + '.bin'))
            torch.save(data['valid']['y'], ('./data/FashionMNIST/y_' + 'valid' + '.bin'))

            data['train']={'x': [],'y': []}
            data['valid']={'x': [],'y': []}
        
            

                
        if s == 'test':
            # for image,target in loader:
            for n, (image,target) in enumerate(loader):
                ### Flatten the (1,32,32) image into (1,1024)
                image = torch.flatten(image)
                image = image.view(1,32,32)
                ### Should give shape (3,32,32)
                image = torch.cat((image,image,image), dim=0)


                data['test']['x'].append(image)
                data['test']['y'].append(target.numpy()[0])
        
            data['test']['x']=torch.stack(data['test']['x']).view(-1,size[0],size[1],size[2])
            data['test']['y']=torch.LongTensor(np.array(data['test']['y'],dtype=int)).view(-1)
            
            torch.save(data['test']['x'], ('./data/FashionMNIST/x_' + 'test' + '.bin'))
            torch.save(data['test']['y'], ('./data/FashionMNIST/y_' + 'test' + '.bin'))
                
            data['test']={'x': [],'y': []}
            
                        
        