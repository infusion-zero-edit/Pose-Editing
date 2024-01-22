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
<<<<<<< HEAD
```
Note: Lamp is generated from 105000.ckpt and other are generated from zero123-xl.ckpt. 

Example Runs:
python task2.py --image ./inputs/office_chair.jpg --class_name "office chair" --azimuth +32 --polar +0 --output ./generated_office_chair.png
python task2.py --image ./inputs/laptop.jpg --class_name "laptop" --azimuth -32 --polar +0 --output ./generated_laptop.png
python task2.py --image ./inputs/lamp.jpg --class_name "lamp" --azimuth -6 --polar +0 --output ./generated_lamp.png
python task2.py --image ./inputs/flower_vase.jpg --class_name "flower vase" --azimuth +12 --polar +0 --output ./generated_flower_vase.png
python task2.py --image ./inputs/chair_1.jpg --class_name "chair_1" --azimuth +32 --polar +0 --output ./generated_chair_1.png
```


=======
>>>>>>> 401c833e7955ebb2e1919f4aa9141a7d198b0311
