import os
from torch.utils.data import Dataset


# we expect the dataset to be in the format: txt file, metadata can be  some hyperparameters for generators, here we just use the prompts
class TextPromptDataset(Dataset):
    def __init__(self, dataset_path=None):
        if dataset_path is not None:
            self.file_path = dataset_path
            with open(self.file_path, 'r') as f:
                self.prompts = [line.strip() for line in f.readlines()]
        else:
            self.prompts =  ["A white polar bear cub wearing sunglasses sits in a meadow with flowers.",
                     "A photorealistic tiny dragon taking a bath in a teacup, coherent, intricate",
                     "A train ride in the monsoon rain in Kerala. With a Koala bear wearing a hat looking out of the window. There is a lot of coconut trees out of the window.",
                     "A planisphere lavalamp glows inside a glass jar buried in sand with swirling mist around it.",
                     "A photograph of a giant diamond skull in the ocean, featuring vibrant colors and detailed textures.",
                     "A still of Doraemon from 'Shaun the Sheep' by Aardman Animation.",
                     "fantasy, pastel, absurdist, photo, tiny teapot house matchbox",
                     "A television made of water that displays an image of a cityscape at night",
                     "harry potter as a skyrim character",
                     "A tall giraffe in a zoo eating branches",
                     "A beautiful award winning picture of a cute cat in front of a dark background. The cat is a cat-peacock hybrid and has a peacock tail and short peacock feathers on the body. fluffy, extremely detailed, stunning, high quality atmospheric lighting",
                     "A yellow cup and two purple cell phones, E-commerce poster, minimalist design",
                     "A photo of four apples",
                     "a futuristic spaceship cockpit with a digital screen displaying 'fuel low' warning, set against the backdrop of a star - studded galaxy, with control panels and futuristic interfaces around.",
                     "vibrant urban alley with a graffiti wall prominently spray-painted 'Street Art Rules', surrounded by colorful tags and murals, under a sunny sky",
                     "A lemon with a McDonald's hat.",
                    "Generate a picture of a blue sports car parked on the road, metal texture",
                    "Daft punk driving through the city at night",
                    "Meat all over the place and blood and a green plant in the middle, cinematic lighting",
                    "The feeling of a mirage in a house, the sunset shining on the lake surface, and the water flowing slowly",
                    "A cozy living room with a painting of a corgi on the wall above a couch and a round coffee table in front of a couch and a vase of flowers on a coffee table",
                    "A tranquil park furnished with rows of benches made of marble",
                    "An anthropomorphic fox man wearing a long coat walking across a glacier, hands in pockets, character illustration by Aaron Miller, Greg Rutkowski, thomas kinkade, Howard Pyle.",
                    "Cars, people, buildings and street lamps on a city street"
                        ]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return  self.prompts[idx], {}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

