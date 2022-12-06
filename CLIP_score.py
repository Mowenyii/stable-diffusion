import os
import clip
import torch
import PIL

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
clip_model, preprocess = clip.load("ViT-L/14", device=device, jit=False)

target_path = "/home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/target_samples"
target_list = os.listdir(target_path)
target_list.sort()

j=1
# for j in range(10):
if j==1:
    ours_path="/home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/ours_lamb1.0/"#+str(j)+"/"
    # ours_path="/home/wenyi_mo/stable-diffusion/FashionIQ_sample/ours_beta0.95/"+str(j)+"/"
    # ours_path="/home/wenyi_mo/stable-diffusion/FashionIQ_sample/ours_beta0.95/7/"
    ours_list=os.listdir(ours_path)
    ours_list.sort()
    # stable_path="/home/wenyi_mo/stable-diffusion/FashionIQ_VAL_sample/stable_samples"
    # stable_list=os.listdir(stable_path)
    # stable_list.sort()






    ours_score=0
    stable_score=0
    for i in range(len(ours_list)):
        ours=os.path.join(ours_path,ours_list[i])
        # stable=os.path.join(stable_path,stable_list[i])
        target=os.path.join(target_path,target_list[i])
        with open(ours, 'rb') as f:
            img_ours = PIL.Image.open(f)
            img_ours = img_ours.convert('RGB')
        # with open(stable, 'rb') as f:
        #     img_stable = PIL.Image.open(f)
        #     img_stable = img_stable.convert('RGB')
        with open(target, 'rb') as f:
            img_target = PIL.Image.open(f)
            img_target = img_target.convert('RGB')
        image_ours = preprocess(img_ours).unsqueeze(0).to(device)
        # image_stable = preprocess(img_stable).unsqueeze(0).to(device)
        image_target = preprocess(img_target).unsqueeze(0).to(device)
        with torch.no_grad():
            features_ours = clip_model.encode_image(image_ours)
            # features_stable = clip_model.encode_image(image_stable)
            features_target = clip_model.encode_image(image_target)
        features_ours /= features_ours.norm(dim=-1, keepdim=True)
        # features_stable /= features_stable.norm(dim=-1, keepdim=True)
        features_target /= features_target.norm(dim=-1, keepdim=True)

        ours_score=ours_score+(features_ours @ features_target.T)#100.0 *
        # stable_score=stable_score+(features_stable @ features_target.T)


    ours_score=ours_score/len(ours_list)
    # stable_score=stable_score/len(ours_list)
    print(j,ours_score)#,stable_score)

