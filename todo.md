tensorboard --logdir=D:\D_code\stable-diffusion\logs


/home/wenyi_mo/stable-diffusion/11-06/14_48ctx.pt


np.fabs((a-b.to(a.device)).cpu().detach().numpy())<1e-5

{"pairid": 29436, "reference": "dev-1053-1-img0", "target_hard": "dev-1053-0-img0", "caption_extend": {"0": "Being a photo of dogs of the same breed", "1": "Remove the collar, make the dog stand with close mouth", "2": "[cr0] Nothing worth mentioning", "3": "Show only grass"}}
cap.ext.rc2.val.json
A photo of a dog chewing on the toy



nohup python train_txt_emb.py --base ./configs/latent-diffusion/one_pic.yaml -t --scale_lr False --gpus 0,1 --max_epochs 100 > ./train_txt.log 2>&1 &

nohup python main.py --base ./configs/latent-diffusion/one_pic.yaml -t --scale_lr False --gpus 0,1 --max_epochs 10 > ./main.log 2>&1 &

python sample.py


python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_VAL/ours_samples /home/wenyi_mo/stable-diffusion/FashionIQ_VAL/target_samples

python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_VAL/stable_samples  /home/wenyi_mo/stable-diffusion/FashionIQ_VAL/target_samples



CUDA_VISIBLE_DEVICES=1 python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_ours_stable/ours_samples /home/wenyi_mo/stable-diffusion/FashionIQ_VAL/target_samples

CUDA_VISIBLE_DEVICES=1 python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_ours_stable/stable_samples  /home/wenyi_mo/stable-diffusion/FashionIQ_VAL/target_samples

python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/ours_samples /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples

python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/stable_samples  /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples



Automatic attribute discovery and characterization from noisy web data

三个参数the path of the JSON file, a directory to which the images will be saved, and the path of the hash file (hash files are included in util/hashes/).

python /home/wenyi_mo/dataset/nlvr/nlvr2/util/download_images.py /home/wenyi_mo/dataset/nlvr/nlvr2/data/test1.json /home/wenyi_mo/dataset/nlvr/nlvr2/test /home/wenyi_mo/dataset/nlvr/nlvr2/util/hashes/test1_hashes.json

每次迭代，随机mask还是固定，以及加时间，mask迭代变小？


#0.5 no att 16.5156

            r       s   
2k  0.5     21.232  21.648
100 0.99    21.2    20.98   (b[:-1])
100 0.99    20.96   21.16   [:int(len(b)*0.5)]
去掉att
100 0.99    20.96   21.16   [:-1]

sample
100 0.99    20.96   21.16   (b[:-1])

基本和SD一样的设置
如果edit文本与输入文本相同
e_t_mid=0.5*e_t+0.5*e_t_edit，结果与SD相同

ddim.py搜edit_beta
edit_beta=0.95，无att inject
e_t_mid=edit_beta*e_t+(1-edit_beta)*e_t_edit
CLIP_score和FID，50step, 优于SD

w/o att inject
stable,49,0.99,21.12,103.2932
ours,49,0.99,21.26,102.6496
w att inject
stable,49,0.99,21.12,103.293
ours,49,0.99,21.16,101.7748

w att inject
stable,1499,0.99,19.84,27.8621
ours,1499,0.99,19.744,27.7416


调比例：
not in np.argsort(b)[int(len(b)*0.5):]
edit_guidance用了最无关的50%个，且没有介词
stable,99,0.99,21.1，91.3132
ours,99,0.99,21.16，90.9437

not in [:int(len(b)*0.5)]edit_guidance用了最相关的50%个，且没有介词
stable,99,0.99,21.1，91.3132
ours,99,0.99,21.14，90.4047

去掉"A photo of clothes " 
not in [:int(len(b)*0.5)]
stable,99,0.99,19.88， 89.4346
ours,99,0.99,19.88， 89.6755

去掉"A photo of clothes " 
not in [:-1]
stable,99,0.99,19.88，89.006
ours,99,0.99,19.83， 88.6984

在上面的基础上去掉stopword
stable,99,0.95,19.88，89.43464
ours,99,0.95,19.83，89.57759

结论：还是要stop

各50%，不用stop
stable,99,0.95,19.88， 89.43464
ours,99,0.95,17.11，95.970

[1:]，c_new用除了相关度最低的，其他是edit_con，不用stop
stable,99,0.95,19.88,89.4346
ours,99,0.95,18.77,92.1902

[:-1]c_new用相关度最高的那个词，其他是edit_con，不用stop
stable,99,0.95,19.88，89.43464
ours,99,0.95,18.5，97.66266

edit_con not in [:-1],不用stop,不用"A photo of clothes "
stable,99,0.95,19.88，89.43464
ours,99,0.95,19.83，89.73881

edit_con not in [:-1],用stop,不用"A photo of clothes "
stable,99,0.95,19.88, 89.4346
ours,99,0.95,19.83, 89.57759


edit_con not in [:-1],用stop,最相关的一个词做edit_con【但可能是is或者a被去掉了】,其他词做att inject
stable,499,0.0,19.792,45.96178
ours,499,0.0,17.216
ours,499,0.1,17.68
ours,499,0.2,18.08
ours,499,0.3,18.464
ours,499,0.4,18.88
ours,499,0.5,19.04
ours,499,0.6,19.248
ours,499,0.7,19.536
ours,499,0.8,19.568
ours,499,0.9,19.744,46.1365
0.95         19.728,46.00039
ours,499,1.0,19.808,45.99814

                CLIP_score  FID
stable          19.792,     45.96178
ours,500,-1.0,  19.808      46.09221
ours,500,-0.75, 19.808      46.03505
ours,500,-0.5,  19.84,      46.04461
ours,500,-0.25, 19.856      45.9642086
ours,500,0.0,   19.824      46.025240
ours,500,0.25,  19.824      45.986850
ours,500,0.5,   19.856      46.067399
ours,500,0.75,  19.904      46.166771
ours,500,1.0,   19.856      46.03818




python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/ours_samples /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples

python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/stable_samples  /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples

```python
import os
from shutil import copyfile
imgspath ="/home/wenyi_mo/stable-diffusion/FashionIQ_sample/ours_beta0.95/"

for i in range(4500):
    file=str(i).zfill(5)+".png"
    new_path=os.path.join(imgspath, str(i%9))
    print(new_path)
    os.makedirs(new_path, exist_ok=True)
    copyfile(imgspath+file,new_path+'/'+file)
```
ls | wc -w

python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_sample/ours_beta0.95/0 /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples
python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_sample/ours_beta0.95/1 /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples
python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_sample/ours_beta0.95/3 /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples
python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_sample/ours_beta0.95/4 /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples
python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_sample/ours_beta0.95/5 /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples
python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_sample/ours_beta0.95/6 /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples
python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_sample/ours_beta0.95/7 /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples
python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_sample/ours_beta0.95/8 /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples
python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_sample/ours_beta-0.25 /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples



python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/stable_samples /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples

python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/ours_lamb0.7 /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples

python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/ours_lamb0.8 /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples

python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/ours_lamb0.9 /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples

python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/ours_lamb0.6 /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples
python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/ours_lamb0.5 /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples
python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/ours_lamb0.4 /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples
python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/ours_lamb0.3 /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples
python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/ours_lamb0.2 /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples
python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/ours_lamb0.1 /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples
python -m pytorch_fid /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/ours_lamb0.0 /home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples


