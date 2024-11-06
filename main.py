import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model.TransUNetv2 import TransUNet
import torch.nn.functional as F
from torchvision.utils import save_image
from utils.config import *
from tqdm import tqdm
from utils.data import MattingDataset
from torch import nn
from utils.metrics import calculate_sad_mse_mad_whole_img,compute_gradient_whole_image,compute_connectivity_loss_whole_image
from utils.matting_crition import loss_gradient_penalty,unknown_l1_loss
from tensorboardX import SummaryWriter
from tqdm import tqdm

# 训练和测试过程
def train_and_evaluate_model(train_loader, test_loader, model, optimizer, epochs, output_dir, test_interval, device):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        # 设置 bar_format 以显示剩余时间
        train_loader_iter = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}",  
                                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Elapsed: {elapsed} | Remaining: {remaining}]")
        
        for batch_idx, (images, alphas, trimap) in enumerate(train_loader_iter):
            optimizer.zero_grad()
            images = images.to(device)
            alphas = alphas.to(device)
            trimap = trimap.to(device)

            # 前向传播
            pred_alpha = model(images,trimap)

            # os.makedirs(os.path.join(output_dir,'train'), exist_ok=True)
            # if epoch==30:
            #     save_image(torch.cat((pred_alpha,alphas,trimap),dim=0), os.path.join(output_dir,'train', f"batch_{batch_idx}.png"))

            # 计算损失
            # loss =  F.mse_loss(pred_alpha, alphas)+F.l1_loss(pred_alpha,alphas)+get_alpha_loss(pred_alpha,alphas,trimap)+1e-6
            # loss =  F.mse_loss(pred_alpha, alphas)+F.l1_loss(pred_alpha,alphas)+get_alpha_loss(pred_alpha,alphas,trimap)+F.cross_entropy(pred_alpha,alphas)+1e-6
            sample_map = torch.zeros_like(trimap)
            sample_map[(trimap >= 0.5) & (trimap < 1)] = 1

            loss = loss_gradient_penalty(sample_map,pred_alpha,alphas)+unknown_l1_loss(sample_map,pred_alpha,alphas)+F.binary_cross_entropy(pred_alpha,alphas)+1e-6
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_loader_iter.set_postfix(loss=loss.item())
            train_loader_iter.set_description(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / (batch_idx + 1):.4f}")

        # 每隔 test_interval 进行测试
        if (epoch + 1) % test_interval == 0:
            model.eval()
            evaluate_and_save(model, test_loader, epoch, output_dir, device)


# 测试模型并保存结果
def evaluate_and_save(model, test_loader, epoch, output_dir, device):
    model.eval()
    test_loss = 0
    total_SAD = 0
    total_MSE = 0
    total_MAD = 0
    total_Conn = 0
    total_Grad = 0

    with torch.no_grad():
        test_loader_iter = tqdm(test_loader, desc=f"Evaluating Epoch {epoch+1}")
        for batch_idx, (images, alphas, trimap) in enumerate(test_loader_iter):
            images = images.to(device)
            alphas = alphas.to(device)
            trimap = trimap.to(device)
            # 前向传播
            pred_alpha = model(images,trimap)
            os.makedirs(os.path.join(output_dir,'test'), exist_ok=True)
            save_image(pred_alpha, os.path.join(output_dir, 'test',f"batch_{batch_idx}.png"))

            # 计算损失和SAD
            # loss =  F.mse_loss(pred_alpha, alphas)+F.l1_loss(pred_alpha,alphas)+get_alpha_loss(pred_alpha,alphas,trimap)+F.cross_entropy(pred_alpha,alphas)+1e-6
            # loss =  F.mse_loss(pred_alpha, alphas)+F.l1_loss(pred_alpha,alphas)+get_alpha_loss(pred_alpha,alphas,trimap)+1e-6

            sample_map = torch.zeros_like(trimap)
            sample_map[(trimap >= 0.5) & (trimap < 1)] = 1

            loss = loss_gradient_penalty(sample_map,pred_alpha,alphas)+unknown_l1_loss(sample_map,pred_alpha,alphas)+F.binary_cross_entropy(pred_alpha,alphas)+1e-6
            
            pred_alpha=pred_alpha.data.cpu().numpy()[0,0,:,:]
            alphas=alphas.data.cpu().numpy()[0,0,:,:]


            SAD, MSE, MAD = calculate_sad_mse_mad_whole_img(pred_alpha, alphas)
            Grad = compute_gradient_whole_image(pred_alpha, alphas)
            Conn = compute_connectivity_loss_whole_image(pred_alpha, alphas)

            test_loss += loss.item()
            total_SAD += SAD.item()
            total_MSE += MSE.item()
            total_MAD += MAD.item()
            total_Grad += Grad.item()
            total_Conn += Conn.item()

            

            test_loader_iter.set_postfix(loss=loss.item(), SAD=SAD.item())

    avg_loss = test_loss / len(test_loader)
    avg_SAD = total_SAD / len(test_loader)
    avg_MSE = total_MSE / len(test_loader)
    avg_MAD = total_MAD / len(test_loader)
    avg_Grad = total_Grad / len(test_loader)
    avg_Conn = total_Conn / len(test_loader)

    print(f"Test Results - Epoch [{epoch+1}]: Loss: {avg_loss}, SAD: {avg_SAD}, MSE: {avg_MSE}, MAD: {avg_MAD}, Grad: {avg_Grad}, Conn: {avg_Conn}")

    writer.add_scalars('test loss',{'test_loss':avg_loss},epoch)
    writer.add_scalars('SAD',{'SAD':avg_SAD},epoch)
    writer.add_scalars('MSE',{'MSE':avg_MSE},epoch)
    writer.add_scalars('Grad',{'Grad':avg_Grad},epoch)
    writer.add_scalars('Conn',{'Conn':avg_Conn},epoch)
    writer.add_scalars('MAD',{'MAD':avg_MAD},epoch)

if __name__ == '__main__':

    writer=SummaryWriter(args.log_dir)

    train_dataset = MattingDataset(
        args.image_dir, 
        args.mask_dir, 
        args.alpha_dir, 
        mode='train'
    )
    # train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size)

    test_dataset = MattingDataset(
        args.image_dir, 
        args.mask_dir, 
        args.alpha_dir, 
        mode='test'
    )
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化模型和优化器
    model = TransUNet(img_dim=args.patch_size,
                    in_channels=3,
                    out_channels=128,
                    head_num=4,
                    mlp_dim=256,
                    block_num=2,
                    patch_dim=16,
                    class_num=1
                    ).to(device)

    # model = nn.DataParallel(model)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=float(args.learning_rate))

    # 开始训练和测试
    train_and_evaluate_model(
        train_loader, 
        test_loader, 
        model, 
        optimizer, 
        args.epochs, 
        args.save_dir, 
        args.test_interval, 
        device  # 传递device
    )
