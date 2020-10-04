# FovealBoosted

This a tool for nuclei segmentation. Published in 

make_foveal.py is for foveal blurred image generation.
unet.py is the file of unet model description.
train.py is the main entrance of training of deep learning system.

First Header | Second Header
------------ | -------------
make_foveal.py | foveal blurred image generation
unet.py | unet model definition
test.py | main entrance of test of deep learning system
train.py | main entrance of training of deep learning system

	rootdir

		--foveal/ *.mat containing foveal blurred images.

		--cells/ *.mat conatining nuclei-level masks.

		--resized_prior.mat shape prior.
  
