from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import random 
# import matplotlib.pyplot as plt
from torchvision import models
from torchsummary import summary

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import gunpowder as gp
import zarr
import math

from torch.utils.tensorboard import SummaryWriter
import skimage
import networkx
import pathlib
from tifffile import imread, imwrite
import tensorboard

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# function to convert 'TRA' channel into cell and frame-wise centroid positions
## Function to extract trajectories from data

base_path = pathlib.Path("/mnt/shared/celltracking/data/cho/")


# read parent-child links from file

epochs = 100
links = np.loadtxt(base_path / "01_GT/TRA" / "man_track.txt", dtype=int)

# read annotated image stack
centroids = np.stack([imread(xi) for xi in sorted((base_path / "01_GT/TRA").glob("*.tif"))])  # images

# extract centroids from annotated image stacks
centers = skimage.measure.regionprops(centroids[0,0,:,:])
tracks = []
for t, frame in enumerate(centroids):
    centers = skimage.measure.regionprops(frame)
    for c in centers:
        tracks.append([c.label, t, int(c.centroid[1]), int(c.centroid[2])])
        
# constructs graph 
tracks = np.array(tracks)
graph = networkx.DiGraph()
for cell_id, t, x, y in tracks:
    graph.add_node((cell_id,t), x=x, y=y, t=t)
    
for cell_id, t in graph.nodes():
    if (cell_id, t+1) in graph.nodes():
        graph.add_edge((cell_id, t), (cell_id,t+1))

for child_id, child_from, _, child_parent_id in links:
    for parent_id, _, parent_to, _ in links:
        if child_parent_id == parent_id:
            graph.add_edge((parent_id, parent_to), (child_id, child_from))
            
# extract trajectories from graph set
tracks = [graph.subgraph(c) for c in networkx.weakly_connected_components(graph) if len(c)>0]

# remove tracks with 0 edges
tracks = [track for track in tracks if len(track.edges)>0]



class getPaired(gp.BatchFilter):

    def __init__(self, raw, raw_shift, tracks, paired=True):
        self.raw = raw
        self.raw_shift = raw_shift
        self.tracks = tracks
        self.paired = paired
    
    # _ref channel array is stored in raw_ref, while second volume in pair will be stored raw_new
    def prepare(self, request):
        # obtain volume coordinates from tracks                
        deps = gp.BatchRequest()
        vol1,vol2 = self.sampler(request)
                
        deps[self.raw] = gp.ArraySpec(roi=gp.Roi(vol1,request[self.raw].roi.get_shape()))
        deps[self.raw_shift] = gp.ArraySpec(roi=gp.Roi(vol2,request[self.raw_shift].roi.get_shape()))

        return deps
    
    # required to inform downstream nodes about new array 
    def process(self, batch, request):
        # create a new batch to hold the new array
        out_batch = gp.Batch()

        # create new array and store it in the batch
        out_batch[self.raw_shift] = batch[self.raw_shift]
        out_batch[self.raw] = batch[self.raw]
        
        #print(f'raw: {batch[self.raw].spec.roi}')
        #print(batch[self.raw_shift].spec.roi)
        
        # make sure that coordinates for batch[raw] and batch[raw_shift] are reset to (0,0,0,0,0)
        out_batch[self.raw].spec.roi = request[self.raw].roi
        out_batch[self.raw_shift].spec.roi = request[self.raw_shift].roi

        # return the new batch
        return out_batch
    
    # select pairs of subvolumes from data
    def sampler(self,request):
        tracks = self.tracks
        paired = self.paired
        # choose connected nodes
        # if self.paired:
        if paired:
            t0 = tracks[np.random.randint(0,len(tracks),1).item()]
            e0 = list(t0.edges)[np.random.randint(len(list(t0.edges)))]
            node0 = t0.nodes[e0[0]]
            node1 = t0.nodes[e0[1]]
            
        # choose random unconnected nodes
        else:
            # randomly choose two tracks and make sure they are not identical
            t0,t1 = np.random.randint(0,len(tracks),2)
            while t0==t1:
                t0,t1 = np.random.randint(0,len(tracks),2)

            #print(f'trackids: {t0,t1}')
            t0 = tracks[t0]
            t1 = tracks[t1]

            # choose random edges from each track
            #print(f'number edges per track{len(list(t0.nodes)),len(list(t1.nodes))}')

            r0 = np.random.randint(0,len(list(t0.nodes))) 
            r1 = np.random.randint(0,len(list(t1.nodes)))

            node0 = t0.nodes[list(t0.nodes)[r0]]
            node1 = t1.nodes[list(t1.nodes)[r1]]
            


        node0_xyt = [node0["x"], node0["y"], node0["t"]]
        node1_xyt = [node1["x"], node1["y"], node1["t"]]

        #print(f'input coord: {node0_xyt,node1_xyt}')

        roi_in = request[self.raw_shift].roi.get_shape()
        #t,z,y,x
        coords_vol0 = (node0_xyt[2],0,node0_xyt[0]-(roi_in[2]/2),node0_xyt[1]-(roi_in[3]/2))
        coords_vol1 = (node1_xyt[2],0,node1_xyt[0]-(roi_in[2]/2),node1_xyt[1]-(roi_in[3]/2))
        #print(f'output coords - vol0: {coords_vol0}, vol1:{coords_vol1}')

        return coords_vol0, coords_vol1
            
        
        
        #specify subvolume size and volume source
volSize = (1,5,64, 64)
coord = (0,0,0,0)
batch_size = 1

zarrdir = '/mnt/shared/celltracking/data/cho/01.zarr'
raw = gp.ArrayKey('raw')
raw_shift = gp.ArrayKey('raw_shift')

# create "pipeline" consisting only of a data source


#Augmentations=

# chose a random source (i.e., sample) from the above
#random_location = gp.RandomLocation()

pipeline_paired = (gp.ZarrSource(
    zarrdir,  # the zarr container
    {raw_shift: 'raw', raw: 'raw'},  # which dataset to associate to the array key
    {raw_shift: gp.ArraySpec(voxel_size=(1,1,1,1), interpolatable=True), raw:gp.ArraySpec(voxel_size=(1,1,1,1), interpolatable=True)}  # meta-information
    )+ gp.Normalize(raw)+gp.Normalize(raw_shift)+ 
    gp.Pad(raw_shift, None) + 
    gp.Pad(raw, None) + gp.IntensityAugment(
    raw,
    scale_min=0.9,
    scale_max=1.1,
    shift_min=-0.1,
    shift_max=0.1,
    ) + gp.NoiseAugment(raw, mode="gaussian")) + gp.IntensityAugment(
    raw_shift,
    scale_min=0.9,
    scale_max=1.1,
    shift_min=-0.1,
    shift_max=0.1,
    ) + gp.NoiseAugment(raw_shift, mode="gaussian") + getPaired(raw,raw_shift,tracks,paired=True) + gp.ElasticAugment(
    [2,10,10],
    [0,2,2],
    [0,0*math.pi/2.0],
    prob_slip=0.05,
    prob_shift=0.05,
    max_misalign=25) + gp.SimpleAugment(transpose_only=[2, 3], mirror_only=[])



pipeline_unpaired = (gp.ZarrSource(
    zarrdir,  # the zarr container
    {raw_shift: 'raw', raw: 'raw'},  # which dataset to associate to the array key
    {raw_shift: gp.ArraySpec(voxel_size=(1,1,1,1), interpolatable=True), raw:gp.ArraySpec(voxel_size=(1,1,1,1), interpolatable=True)}  # meta-information
    )+ gp.Normalize(raw)+gp.Normalize(raw_shift)+ 
    gp.Pad(raw_shift, None) + 
    gp.Pad(raw, None) + gp.IntensityAugment(
    raw,
    scale_min=0.9,
    scale_max=1.1,
    shift_min=-0.1,
    shift_max=0.1,
    ) + gp.NoiseAugment(raw, mode="gaussian") + gp.IntensityAugment(
    raw_shift,
    scale_min=0.9,
    scale_max=1.1,
    shift_min=-0.1,
    shift_max=0.1,
    ) + gp.NoiseAugment(raw_shift, mode="gaussian")) + getPaired(raw,raw_shift,tracks,paired=False)  + gp.ElasticAugment(
    [2,10,10],
    [0,2,2],
    [0,0*math.pi/2.0],
    prob_slip=0.05,
    prob_shift=0.05,
    max_misalign=25) + gp.SimpleAugment(transpose_only=[2, 3], mirror_only=[]) 

pipeline_paired += gp.PreCache(num_workers=3) 
pipeline_unpaired += gp.PreCache(num_workers=3)
#pipeline_paired += gp.Stack(batch_size)
#pipeline_unpaired += gp.Stack(batch_size)

# specify request
request = gp.BatchRequest()
request[raw] = gp.Roi(coord, volSize)
request[raw_shift] = gp.Roi(coord, volSize)

gp.ArraySpec()
# build the pipeline...
with gp.build(pipeline_paired), gp.build(pipeline_unpaired):

  # ...and request a batch
  batch = pipeline_unpaired.request_batch(request)
  
#show the content of the batch
#print(f"batch returned: {batch}")

# plot first slice of volume
# fig, axs = plt.subplots(1,2)
# print(batch[raw].data.shape)
# axs[0].imshow(np.flipud(batch[raw].data[0,0,:,:]))
# axs[1].imshow(np.flipud(batch[raw_shift].data[0,0,:,:]))



class Vgg3D(torch.nn.Module):

    def __init__(self, input_size, output_classes, downsample_factors, fmaps=12):

        super(Vgg3D, self).__init__()

        self.input_size = input_size
        self.downsample_factors = downsample_factors
        self.output_classes = 2

        current_fmaps, h, w, d = tuple(input_size)
        current_size = (h, w,d)

        features = []
        for i in range(len(downsample_factors)):

            features += [
                torch.nn.Conv3d(current_fmaps,fmaps,kernel_size=3,padding=1),
                torch.nn.BatchNorm3d(fmaps),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv3d(fmaps,fmaps,kernel_size=3,padding=1),
                torch.nn.BatchNorm3d(fmaps),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool3d(downsample_factors[i])
            ]

            current_fmaps = fmaps
            fmaps *= 2

            size = tuple(
                int(c/d)
                for c, d in zip(current_size, downsample_factors[i]))
            check = (
                s*d == c
                for s, d, c in zip(size, downsample_factors[i], current_size))
            assert all(check), \
                "Can not downsample %s by chosen downsample factor" % \
                (current_size,)
            current_size = size

        self.features = torch.nn.Sequential(*features)

        classifier = [
            torch.nn.Linear(current_size[0] *current_size[1]*current_size[2] *current_fmaps,4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096,4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096,output_classes)
        ]

        self.classifier = torch.nn.Sequential(*classifier)
    
    def forward(self, raw):

        # add a channel dimension to raw
        # shape = tuple(raw.shape)
        # raw = raw.reshape(shape[0], 1, shape[1], shape[2])
        
        # compute features
        f = self.features(raw)
        f = f.view(f.size(0), -1)
        
        # classify
        y = self.classifier(f)

        return y
    
    
# class ContrastiveLoss(nn.Module):
#     "Contrastive loss function"

#     def __init__(self, margin=2.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, output1, output2, label):
#         euclidean_distance = F.pairwise_distance(output1, output2)
#         loss_contrastive = torch.mean(
#             (1 - label) * torch.pow(euclidean_distance, 2)
#             + (label)
#             * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
#         )

#         return loss_contrastive
    
    
input_size = (1, 64, 64, 5)
downsample_factors =[(2, 2, 1), (2, 2, 1), (2, 2, 1), (2, 2, 1)];
output_classes = 12

# create the model to train
model = Vgg3D(input_size, output_classes,  downsample_factors = downsample_factors)
model = model.to(device)

# summary(model, input_size)


from tqdm import tqdm

def train(tb_logger = None, log_image_interval = 10):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    loss=[] 
    counter=[]
    with gp.build(pipeline_paired), gp.build(pipeline_unpaired):
        for epoch in tqdm(range(epochs)):
            epoch_loss = 0
            epoch_loss_pos = 0
            epoch_loss_neg = 0
            for x in range(2):
                unpaired = pipeline_unpaired.request_batch(request)
                yu = -1
                
                paired = pipeline_paired.request_batch(request)
                yp = 1
                
                unpaired1 = unpaired[raw].data
                unpaired2 = unpaired[raw_shift].data
                paired1 = paired[raw].data
                paired2 = paired[raw_shift].data
                
                unpaired1 = np.reshape(unpaired1, (batch_size,64, 64, 5))
                unpaired2 = np.reshape(unpaired2, (batch_size,64, 64, 5))
                paired1 = np.reshape(paired1, (batch_size,64, 64, 5))
                paired2 = np.reshape(paired2, (batch_size,64, 64, 5))
                
                # unpaired1 = np.reshape(unpaired1, (batch_size,16, 16, 5))
                # unpaired2 = np.reshape(unpaired2, (batch_size,16, 16, 5))
                # paired1 = np.reshape(paired1, (batch_size,16, 16, 5))
                # paired2 = np.reshape(paired2, (batch_size,16, 16, 5))
                
                unpaired1 = np.expand_dims(unpaired1, axis =0)
                unpaired2 = np.expand_dims(unpaired2, axis=0)
                paired1 = np.expand_dims(paired1, axis =0)
                paired2 = np.expand_dims(paired2, axis=0)
                print('5')
                unpaired1 = torch.from_numpy(unpaired1).to(device).float()
                unpaired2 = torch.from_numpy(unpaired2).to(device).float()
                yu = torch.from_numpy(np.array([yu])).to(device).float()
                
                paired1 = torch.from_numpy(paired1).to(device).float()
                paired2 = torch.from_numpy(paired2).to(device).float() 
                yp = torch.from_numpy(np.array([yp])).to(device).float()

                optimizer.zero_grad()
                
                predp1 = model(paired1)
                predp2 = model(paired2)
                predu1 = model(unpaired1)
                predu2 = model(unpaired2)
                #print(model(unpaired1).shape)
                #print(predp1.shape)
                
                # predp1, predp2 = model(paired1, paired2)
                # predu1, predu2 = model(unpaired1, unpaired2)

                #loss = loss_function(pred, y)
                
                print(predp1.shape)

                loss_contrastivep = loss_function(predp1,predp2,yp)
                loss_contrastiveu = loss_function(predu1,predu2,yu)

                loss_contrastivep.backward()
                loss_contrastiveu.backward()
                optimizer.step()    
                epoch_loss_pos += loss_contrastivep
                epoch_loss_neg += loss_contrastiveu
                epoch_loss += loss_contrastivep + loss_contrastiveu
                
                if tb_logger is not None:
                    step = epoch * 10 + x
                    tb_logger.add_scalar(
                        tag="positive_loss", scalar_value=epoch_loss_pos.item(), global_step=step
                    )
                    tb_logger.add_scalar(
                        tag="negative_loss", scalar_value=epoch_loss_neg.item(), global_step=step
                    )
                    tb_logger.add_scalar(
                        tag="total_loss", scalar_value=epoch_loss.item(), global_step = step
                    )
                    # check if we log images in this iteration
                    # if step % log_image_interval == 0:
                    #     tb_logger.add_images(
                    #         tag="in_unpaired1", img_tensor=unpaired1.to("cpu"), global_step=step
                    #     )
                    #     tb_logger.add_images(
                    #         tag="in_unpaired2", img_tensor=unpaired2.to("cpu"), global_step=step
                    #     )
                    #     tb_logger.add_images(
                    #         tag="in_paired1", img_tensor=paired1.to("cpu"), global_step=step
                    #     )
                    #     tb_logger.add_images(
                    #         tag="in_paired2", img_tensor=paired2.to("cpu"), global_step=step
                    #     )


            print(f"epoch {epoch}, total_loss = {epoch_loss}, positive_loss={epoch_loss_pos}, negative_loss={epoch_loss_neg}")
        
    return model
logger = SummaryWriter()
model = train(tb_logger = logger)
