import sys
#from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import numpy as np
import time

from cyclegan.options.test_options import TestOptions
from cyclegan.models import create_model
from cyclegan.util.util import save_image, tensor2im

class cycleGANsim2real():
    def __init__(self, ckpt_name):
        # simulate command line flags (required by cyclegan)
        sys.argv=[]
        sys.argv.append("")
        sys.argv.append("--dataroot")
        sys.argv.append("")
        sys.argv.append("--name")
        sys.argv.append(ckpt_name)
        sys.argv.append("--model")
        sys.argv.append("test")
        sys.argv.append("--no_dropout")
        sys.argv.append("--checkpoints_dir")
        sys.argv.append("./cyclegan/checkpoints") # final checkpoint path: "cyclegan/checkpoints/"+ckpt_name 
        print("sys.argv:",sys.argv)
        opt = TestOptions().parse()  # get test options
        # hard-code some parameters for test
        opt.num_threads = 0   # test code only supports num_threads = 1
        opt.batch_size = 1    # test code only supports batch_size = 1
        opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
        self.model = create_model(opt)
        self.model.setup(opt)
        self.preprocess = transforms.Compose([
        transforms.ToTensor()
        ])  

    def transform_image(self, image):
        # Load input
        image = self.preprocess(image).unsqueeze_(0) # add new dim at position 0
        data={"A":image, 'A_paths':""}
        self.model.set_input(data)  # unpack data from data loader
        self.model.test() # run inference       
        visuals = self.model.get_current_visuals() # get image results
        fake = tensor2im(visuals["fake"]) # Convert image to numpy array
        # Save image
        #save_image(fake, './fake_B.jpg')
        return fake