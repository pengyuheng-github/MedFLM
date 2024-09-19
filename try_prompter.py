import torch
from minigpt4.models import load_model, load_model_and_preprocess
from PIL import Image

# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
raw_image = Image.open("merlion.png").convert("RGB")


def try_prompter():
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="medgpt_nclass", model_type="pretrain_vicuna0", device=device
    )

    image1 = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    image2 = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    image = torch.cat((image1,image2),0)
    text = "<Img><ImageHere></Img> Describe this image in detail."

    inp = {"image":image, "question":["Describe this image in detail."]*2,"answer":["america sb","china nb class,hhh"]}
    print(inp)
    forward = model.forward(inp, reduction='none')
    print(forward)

    # forward = model.multi_select(images=image, texts=["yes or no"],answers=["no","banana","yes"])
    # print(forward)

try_prompter()

