import datetime
import os
import time

import torch
from torch import nn
import numpy as np
from torchvision.utils import save_image
import random
from dataloader import get_loader
from model import CXLoss, DiscriminatorWithClassifier, GeneratorStyle, myDecoder
from options import get_parser
from vgg_cx import VGG19_CX
import torchvision.transforms as T
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(opts):
    # Dirs
    log_dir = os.path.join("experiments", opts.experiment_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoint")
    samples_dir = os.path.join(log_dir, "samples")
    logs_dir = os.path.join(log_dir, "logs")

    # Loss criterion
    criterion_bce = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_GAN = torch.nn.MSELoss().to(device)
    criterion_pixel = torch.nn.L1Loss().to(device)
    criterion_ce = torch.nn.CrossEntropyLoss().to(device)
    criterion_attr = torch.nn.MSELoss().to(device)

    # CX Loss
    if opts.lambda_cx > 0:
        criterion_cx = CXLoss(sigma=0.5).to(device)
        vgg19 = VGG19_CX().to(device)
        vgg19.load_model('vgg19-dcbb9e9d.pth')
        vgg19.eval()
        vgg_layers = ['conv3_3', 'conv4_2']

    # Path to data
    image_dir = os.path.join(opts.data_root, opts.dataset_name, "image")
    attribute_path = os.path.join(opts.data_root, opts.dataset_name, "attributes.txt")

    # Dataloader
    train_dataloader = get_loader(image_dir, attribute_path,
                                  dataset_name=opts.dataset_name,
                                  image_size=opts.img_size,
                                  n_style=opts.n_style,
                                  batch_size=opts.batch_size, binary=False)
    test_dataloader = get_loader(image_dir, attribute_path,
                                 dataset_name=opts.dataset_name,
                                 image_size=opts.img_size,
                                 n_style=opts.n_style, batch_size=8,
                                 mode='test', binary=False)

    # Model
    generator = GeneratorStyle(n_style=opts.n_style, attr_channel=opts.attr_channel,
                               style_out_channel=opts.style_out_channel,
                               n_res_blocks=opts.n_res_blocks,
                               attention=opts.attention)
    discriminator = DiscriminatorWithClassifier()
    #my
    classifier = myDecoder()
    # Attrbute embedding
    # attribute: N x 37 -> N x 37 x 64
    attribute_embed = nn.Embedding(opts.attr_channel, opts.attr_embed)
    # unsupervise font num + 1 dummy id (for supervise)
    attr_unsuper_tolearn = nn.Embedding(opts.unsuper_num+1, opts.attr_channel)  # attribute intensity

    if opts.multi_gpu:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        attribute_embed = nn.DataParallel(attribute_embed)
        attr_unsuper_tolearn = nn.DataParallel(attr_unsuper_tolearn)
        #
        classifier = nn.DataParallel(classifier)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    attribute_embed = attribute_embed.to(device)
    attr_unsuper_tolearn = attr_unsuper_tolearn.to(device)
    #my
    classifier = classifier.to(device)

    # Discriminator output patch shape
    patch = (1, opts.img_size // 2**4, opts.img_size // 2**4)

    # optimizers
    optimizer_G = torch.optim.Adam([
        {'params': generator.parameters()},
        {'params': attr_unsuper_tolearn.parameters(), 'lr': 1e-3},
        {'params': attribute_embed.parameters(), 'lr': 1e-3}],
        lr=opts.lr, betas=(opts.b1, opts.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opts.lr, betas=(opts.b1, opts.b2))
    #
    optimizer_C = torch.optim.Adam(classifier.parameters(), lr=opts.lr, betas=(opts.b1, opts.b2))

    # Resume training
    if opts.init_epoch > 1:
        gen_file = os.path.join(checkpoint_dir, f"G_{opts.init_epoch}.pth")
        attr_unsuper_file = os.path.join(checkpoint_dir, f"attr_unsuper_embed_{opts.init_epoch}.pth")
        attribute_embed_file = os.path.join(checkpoint_dir, f"attribute_embed_{opts.init_epoch}")
        dis_file = os.path.join(checkpoint_dir, f"D_{opts.init_epoch}.pth")
        cla_file_file = os.path.join(checkpoint_dir, f"classifier_{epoch}.pth")

        generator.load_state_dict(torch.load(gen_file))
        attr_unsuper_tolearn.load_state_dict(torch.load(attr_unsuper_file))
        attribute_embed.load_state_dict(torch.load(attribute_embed_file))
        discriminator.load_state_dict(torch.load(dis_file))
        classifier.load_state_dict(torch.load(cla_file_file))

    prev_time = time.time()
    logfile = open(os.path.join(log_dir, "loss_log.txt"), 'w')
    val_logfile = open(os.path.join(log_dir, "val_loss_log.txt"), 'w')

    attrid = torch.tensor([i for i in range(opts.attr_channel)]).to(device)
    attrid = attrid.view(1, attrid.size(0))
    attrid = attrid.repeat(opts.batch_size, 1)
    
    #attribute_embed.load_state_dict(torch.load("attribute_embed_200.pth"))
    #attr_unsuper_tolearn.load_state_dict(torch.load("attr_unsuper_embed_200.pth"))
    #generator.load_state_dict(torch.load("G_200.pth"))
    #classifier.load_state_dict(torch.load("classifier_200.pth"))
    #discriminator.load_state_dict(torch.load("D_200.pth"))

    for epoch in range(opts.init_epoch, opts.n_epochs+1):
        for batch_idx, batch in enumerate(train_dataloader):
            img_A = batch['img_A'].to(device)
            attr_A_data = batch['attr_A'].to(device)
            fontembd_A = batch['fontembed_A'].to(device)
            label_A = batch['label_A'].to(device)
            charclass_A = batch['charclass_A'].to(device)
            styles_A = batch['styles_A'].to(device)

            img_B = batch['img_A'].to(device)
            attr_B_data = batch['attr_A'].to(device)
            fontembd_B = batch['fontembed_A'].to(device)
            label_B = batch['label_A'].to(device)
            charclass_B = batch['charclass_A'].to(device)

            valid = torch.ones((img_A.size(0), *patch)).to(device)
            fake = torch.zeros((img_A.size(0), *patch)).to(device)

            # Construct attribute
            attr_raw_A = attribute_embed(attrid)
            attr_raw_B = attribute_embed(attrid)

            attr_A_embd = attr_unsuper_tolearn(fontembd_A)
            attr_A_embd = attr_A_embd.view(attr_A_embd.size(0), attr_A_embd.size(2))
            attr_A_embd = torch.sigmoid(3*attr_A_embd)  # convert to [0, 1]
            attr_A_intensity = label_A * attr_A_data + (1 - label_A) * attr_A_embd

            attr_A_intensity_u = attr_A_intensity.unsqueeze(-1)
            attr_A = attr_A_intensity_u * attr_raw_A

            attr_B_embd = attr_unsuper_tolearn(fontembd_B)
            attr_B_embd = attr_B_embd.view(attr_B_embd.size(0), attr_B_embd.size(2))
            attr_B_embd = torch.sigmoid(3*attr_B_embd)  # convert to [0, 1]
            attr_B_intensity = label_B * attr_B_data + (1 - label_B) * attr_B_embd

            attr_B_intensity_u = attr_B_intensity.unsqueeze(-1)
            attr_B = attr_B_intensity_u * attr_raw_B

            #delta_intensity = attr_B_intensity - attr_A_intensity
            #delta_attr = attr_B - attr_A
            #delta_intensity = torch.zeros(64,37).to(device)
            #delta = torch.rand(1,37)
            my_emb0 = torch.Tensor(np.random.choice([0,1],(1, 1))).to(device)
            my_emb1 = my_emb0.repeat(64,1)
            delta_intensity = my_emb0.repeat(64,37)
            
            #for thei in range(64):
            #    mydelta = torch.rand(1,37)
            #    for thej in range(37):
            #        delta_intensity[thei,thej] =  mydelta[0,thej]
            attr_A_intensity_u = delta_intensity.unsqueeze(-1)
            delta_attr = attr_A_intensity_u * attr_raw_A

            # Forward G and D
            fake_B, content_logits_A = generator(img_A, styles_A, delta_intensity, delta_attr)

            pred_fake, real_A_attr_fake, fake_B_attr_fake = discriminator(img_A, fake_B, charclass_A, attr_A_intensity+delta_intensity)
            pred_real, real_A_attr, real_B_attr = discriminator(img_A, img_A, charclass_A, attr_A_intensity+delta_intensity)
            
            loss_pixel = opts.lambda_l1 * criterion_pixel(fake_B, img_B)
            if opts.lambda_cx > 0:
                vgg_fake_B = vgg19(fake_B)
                vgg_img_B = vgg19(img_B)
            #loss_GAN = opts.lambda_GAN * criterion_bce(pred_fake, valid)
            loss_GAN = opts.lambda_GAN * criterion_GAN(pred_fake, valid)
            
            loss_attr = torch.zeros(1).to(device)
            if opts.dis_pred:
                loss_attr += opts.lambda_attr * criterion_attr(attr_A_intensity, real_A_attr_fake)
                loss_attr += opts.lambda_attr * criterion_attr(attr_A_intensity+delta_intensity, fake_B_attr_fake)
            
            # CX loss
            loss_CX = torch.zeros(1).to(device)
            if opts.lambda_cx > 0:
                for l in vgg_layers:
                    cx = criterion_cx(vgg_img_B[l], vgg_fake_B[l])
                    loss_CX += cx * opts.lambda_cx
            loss_G = loss_pixel + loss_CX + loss_attr
            #loss_G = loss_GAN + loss_pixel + loss_CX + loss_attr

            optimizer_G.zero_grad()
            loss_G.backward(retain_graph=True)
            optimizer_G.step()
            
            loss_fake = criterion_GAN(pred_fake, fake)
            loss_real = criterion_GAN(pred_real, valid)
            loss_D =  opts.lambda_GAN * ( loss_fake + loss_real )

            optimizer_D.zero_grad()
            loss_D.backward(retain_graph=True)
            optimizer_D.step()
            
            
            [myline,myrow] = fake_B_attr_fake.size()
            #accuracy = 0
            #for i in range(myline):
                #if delta_intensity_after[i,36] > 0:
                #        accuracy = accuracy + 1
            #    if real_A_attr_fake[i,34] - fake_B_attr_fake[i,34] > 0:
            #        accuracy = accuracy + 1
            #    if fake_B_attr_fake[i,36] - real_A_attr_fake[i,36] > 0:
            #        accuracy = accuracy + 1
            #accrate = accuracy/(myline*2)
            
            batches_done = (epoch - opts.init_epoch) * len(train_dataloader) + batch_idx
            batches_left = (opts.n_epochs - opts.init_epoch) * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left*(time.time() - prev_time))
            prev_time = time.time()
            transform = []
            transform.append(T.Resize(64))
            transform.append(T.ToTensor())
            transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
            transform = T.Compose(transform)
            myB = torch.zeros(64,3,64,64).to(device)
            
            for i in range(opts.batch_size):
                img_sample =  fake_B[i,:,:,:]
                save_file = os.path.join(logs_dir, f"epoch_{epoch}_batch_{batches_done}_fake_B_{i}.png")
                save_image(img_sample, save_file, nrow=1, normalize=True)
                fake = Image.open(save_file).convert('RGB')
                myB[i] = transform(fake)
                os.remove(save_file)
                       
            myB = myB.to(device)
            #load_B_attr = classifier(myB)
            #load_A_attr = classifier(img_A)
            #loss_class = opts.lambda_attr * criterion_attr(attr_A_intensity+delta_intensity, load_B_attr)
            #loss_class += opts.lambda_attr * criterion_attr(load_A_attr, attr_A_intensity)
            load_B_attr = classifier(myB)
            loss_C = opts.lambda_attr * criterion_attr(my_emb1, load_B_attr)
            optimizer_C.zero_grad()
            loss_C.backward(retain_graph=True)
            optimizer_C.step()
            
            accuracy = 0
            for i in range(myline):
                #if load_A_attr[i,34] - load_B_attr[i,34] > 0:
                #    accuracy = accuracy + 1
                if my_emb1[i,0] == 0:
                    if load_B_attr[i,0] <0.5:
                        accuracy += 1
                else:
                    if load_B_attr[i,0] >=0.5:
                        accuracy += 1
            accrate = accuracy/(myline)#*2)

            message = (
                f"Epoch: {epoch}/{opts.n_epochs}, Batch: {batch_idx}/{len(train_dataloader)},  "
                f"G loss: {loss_G.item():.6f}, "
                f"D loss: {loss_D.item():.6f}, "
                f"acc:{accrate:.6f}"
                f"loss_C: {loss_C.item(): .6f}"
            )

            print(message)
            logfile.write(message + '\n')
            logfile.flush()

            if batches_done % opts.log_freq == 0:
                img_sample = torch.cat((img_A.data, fake_B.data, img_B.data), -2)
                save_file = os.path.join(logs_dir, f"epoch_{epoch}_batch_{batches_done}.png")
                save_image(img_sample, save_file, nrow=8, normalize=True)



        if opts.check_freq > 0 and epoch % 5 == 0:
            gen_file_file = os.path.join(checkpoint_dir, f"G_{epoch+0}.pth")
            attribute_embed_file = os.path.join(checkpoint_dir, f"attribute_embed_{epoch+0}.pth")
            attr_unsuper_embed_file = os.path.join(checkpoint_dir, f"attr_unsuper_embed_{epoch+0}.pth")
            dis_file_file = os.path.join(checkpoint_dir, f"D_{epoch+0}.pth")
            #
            cla_file_file = os.path.join(checkpoint_dir, f"classifier_{epoch+0}.pth")

            torch.save(generator.state_dict(), gen_file_file)
            torch.save(attribute_embed.state_dict(), attribute_embed_file)
            torch.save(attr_unsuper_tolearn.state_dict(), attr_unsuper_embed_file)
            torch.save(discriminator.state_dict(), dis_file_file)
            torch.save(classifier.state_dict(), cla_file_file)



def main():
    parser = get_parser()
    opts = parser.parse_args()
    opts.unsuper_num = 968

    os.makedirs("experiments", exist_ok=True)

    if opts.phase == 'train':
        # Create directories
        log_dir = os.path.join("experiments", opts.experiment_name)
        os.makedirs(log_dir, exist_ok=False)  # False to prevent multiple train run by mistake
  
        os.makedirs(os.path.join(log_dir, "checkpoint"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "results"), exist_ok=True)
        
        os.makedirs(os.path.join(log_dir, "logs"), exist_ok=True)

        print(f"Training on experiment {opts.experiment_name}...")
        # Dump options
        with open(os.path.join(log_dir, "opts.txt"), "w") as f:
            for key, value in vars(opts).items():
                f.write(str(key) + ": " + str(value) + "\n")
        train(opts)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
