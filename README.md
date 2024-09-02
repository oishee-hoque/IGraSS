# IGraSS-Iterative-Graph-constrained-Semantic-Segmentation-From-Sattelite-Imagery
### Usage

#### Dataset




#### Training IGraSS

To train a IGraSS framework to get a refined ground truth:

    /Framework/run_framework.py --iterations 5 --process_type f --model_type resnet --output_path /output_path/ --from_scratch --dilation --k 4 --R 150 --th 0.5 --r_th 0.1 --epoch 10

To see all optional arguments for training:

    /Framework/run_framework.py -h
### Testing the Model

### Visualization
