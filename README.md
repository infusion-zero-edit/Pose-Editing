# Problem Statement: Edit pose of an object in a scene

Recent advancement in generative AI has led to a development of a lot of creative workflows. One
such workflow is to use generative AI techniques for editing product photographs after they have
been shot at a studio, for example to polish the scene for displaying the product on an e-commerce
website. One such post-production editing requirement could be editing the pose of the object by
rotating it within the same scene.

This problem statement involves two tasks - for the eventual goal of developing technology for a
user-friendly pose edit functionality. The first task is to segment an object (defined by a user given
class prompt) in a given scene. This enables the ‘user-friendly’ part of the problem statement. The
second task is to edit the pose of the object by taking user poses (e.g. Azimuth +10 degrees, Polar -5
degrees). The final generated scene should look realistic and composite.

Tasks:
1. Task1. Take the input scene and the text prompt from the command line argument and outputs an image with a red mask on all pixels where the object (denoted in the text prompt) was present
2. Task2. The second task is to change the pose of the segmented object by the relative angles given by the user. You can use a consistent direction as positive azimuth and polar angle change and mention what you used.

The generated image:
1. Should preserve the scene (background)
2. Should adhere to the relative angles given by the user

## Usage
```
pip install -r requirements.txt
pip install -e taming-transformers/
pip install -e CLIP/
pip install -e GroundingDINO/
```
Create folder checkpoints 
```
mkdir checkpoints
wget https://cv.cs.columbia.edu/zero123/assets/105000.ckpt
wget https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt
```
Example Run Task-1
```
python task1.py --image ./inputs/office_chair.jpg --class_name "office chair" --output ./generated_office_chair.png
```
Example Run Task-2
```
python task2.py --image ./inputs/office_chair.jpg --class_name "office chair" --azimuth +72 --polar +0 --output ./generated_office_chair.png
```
We have used zero123-xl.ckpt and 105000.ckpt interchangeably to get best results which is stored in the folder outputs. For example lamp example gives best output with 105000.ckpt however zero123-xl.ckpt does not able to produce lamp and it gives random image. While in case of office chair the checkpoint 105000.ckpt produce distorted image while zero123-xl.ckpt produces better image.

## Results
```
Note: Lamp is generated from 105000.ckpt and other are generated from zero123-xl.ckpt. 

Example Runs:
python task2.py --image ./inputs/office_chair.jpg --class_name "office chair" --azimuth +32 --polar +0 --output ./generated_office_chair.png
python task2.py --image ./inputs/laptop.jpg --class_name "laptop" --azimuth -32 --polar +0 --output ./generated_laptop.png
python task2.py --image ./inputs/lamp.jpg --class_name "lamp" --azimuth -6 --polar +0 --output ./generated_lamp.png
python task2.py --image ./inputs/flower_vase.jpg --class_name "flower vase" --azimuth +12 --polar +0 --output ./generated_flower_vase.png
python task2.py --image ./inputs/chair_1.jpg --class_name "chair_1" --azimuth +32 --polar +0 --output ./generated_chair_1.png
```
| Input                               | SAM Output                          | Rotated object                     |
| ----------------------------------- | ----------------------------------- |----------------------------------- |
| ![chair](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/ad6dc84b-24b5-4e61-a164-4febacc4f7d4) |  ![generated_mask_chair](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/3a06e97e-912b-4618-b59a-42747cfff772) | ![generated_chair](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/5e3ade2b-fec0-4c55-a37f-cd275c871f1e) |
| ![chair(1)](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/b829d968-5395-4d1e-b539-3e55ad0556b3) | ![generated_mask_chair_1](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/7fc8a82f-ce75-406b-8f5d-46258616ba6d) | ![generated_chair_1](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/0356baa8-d11b-4b79-afbb-0ec6285567f9) |
| ![sofa](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/addf01ab-51c9-4575-9343-a206e7515052) | ![generated_mask_sofa](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/77818f32-3e8a-486c-b68c-2012de8dc910) | ![generated_sofa](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/47a70068-7b43-497a-a043-ae7b75dad236) | 
| ![laptop](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/378460f4-66e8-4bd4-a009-77ec6b3cb64c) | ![generated_mask_laptop](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/bc05ef20-d713-4f6a-8351-0f4b2ac0aefc) | ![generated_laptop](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/0555e94d-2a50-4f8a-af8b-6d6ef563d68b) | 
| ![lamp](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/e309b680-e77a-472b-b136-e7aa78b187ec)| ![generated_mask_lamp](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/40fbdd47-5856-4236-bf11-cb187c963d86)| ![generated_lamp](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/0eadb2a4-af6a-42dc-a914-b1418bb299b2) | 
| ![flower vase](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/05a1bea0-ea82-4aa2-b557-2e54610ce15c) | ![generated_mask_flower_vase](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/36bccc9d-e9c2-44d9-a69d-53c18a76e9fa) | ![generated_flower_vase](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/025e5bed-d91c-47eb-b9b0-7e48a5387db5) |
| ![office chair](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/0f02d6b8-025c-427b-af11-d9b9885624af) | ![generated_mask_office_chair](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/d942f099-3ebc-428a-9497-fa2d395869a0)|![generated_office_chair](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/68d819e2-ee5e-4d10-b46a-c499a5ccabee)|
| ![table](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/5e3bf784-6e81-473c-b3a2-8d32e6eb4304) | ![generated_mask_table](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/81c9fa9e-0348-438f-bd34-69cf7f1e9548)| ![generated_table](https://github.com/infusion-zero-edit/Pose-Editing/assets/122880654/99ba7cca-6015-47c4-a74e-ddabb98c099b)|

## Success and Failures

**Success**: Clearly the foundation model SAM is able to accurately generates mask for the object specified in the text prompt and hence gives us the flexibility to extract the object and to use it for further processing i.e. rotation in this case. Not only SAM but the inpainting model LaMa also generates the background images which are realistic giving us the seamless flexibility to paste the rotated object without caring for the background distortion. 

**Failures**: Zero123 although trainied on ObjaVerse Dataset is not yet generalizable for all the objects for example lamp is only generated by 105000.ckpt rather that XL version zero123-xl.ckpt. Also all the angles are not yet generalizable. There are other project Syncdreamer, HarmonyView (improvement of Syncdreamer), Wonder3D etc has been introduced but they are also not generalizable for all the objects. This is due to the dataset scale required for training foundational model, in 3D this is a very large issue since objaverse dataset with only 800K objects amounts to dataset size of around 1.8TB and training requires very large infrastructure. This is an ongoing research to generate novel views with Stable Diffusion knowledge so that we can utilise the large scale training of Stable Diffusion. Infact, Wonder3D trained on top of StableDiffusionImgtoImg to utilise the image transformation capability of stable diffusion which is trained on very large scale datasets. Not only this we have to extract the depth map of the background so that we can paste the rotated object seamlessly on background which can give the realistic feeling currently from output images after rotation is not looking realistic and hence it requires improvement so that they can look more realistic. With respect to angle of rotation in 3D we can gather the displacement in x and y corrdinates if the depth map of background image is known and hence we can place the image in a realistic manner. Further, for the domain specific images we can fine-tune on the in-domain images to improve the generation of rotated images. Further to make them photorealistic we can use CodeFormer for Super-Resolution to further enhance the photorealism. 

















