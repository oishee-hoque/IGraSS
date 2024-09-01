import argparse

from generate_data import *
from resunet_train import *
from run_single_process import *


def run_process(args):
    

    for i in range(args.iterations):
        print(f'\n\n\n\n\nIterations: {i}\n\n\n\n\n')
        
        # args.output_path = f'{args.output_path}it{i}_R{args.R}_th{args.th}_rth{args.r_th}_{args.k}dil_{args.model_type}/'
        output_path = f'{args.output_path}R{args.R}_th{args.th}_rth{args.r_th}_{args.k}dil_{args.model_type}_i{args.epochs}/'
        args.subfolder_prefix = f'R{args.R}_th{args.th}_rth{args.r_th}_{args.k}dil_{args.model_type}_i{args.epochs}'
        args.prefix = f'it{i}_R{args.R}_th{args.th}_rth{args.r_th}_{args.k}dil_{args.model_type}_i{args.epochs}'
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    
        if args.process_type == 'd' or args.process_type == 'f' or args.process_type == 't':
            image_path, mask_path = generate_dataset_set(args.image_path,args.gt_path,output_path,args.subfolder_prefix,args.patch_size,args.dilation,args.k)
            if args.process_type == 'd':
                return
            
            args.img_folder_name = image_path
            args.mask_folder_name = mask_path


        if args.process_type == 't' or args.process_type == 'f':
            
            if args.from_scratch == False:
                assert args.pretrained_weights != ""

            if args.model_type.lower() == 'resunet' or args.model_type.lower() == 'resnet' or args.model_type.lower() == 'deeplabv3+':
                trained_weights_path = run_resunet(DATA_DIR = args.data_dir,
                            img_folder_name = args.img_folder_name,
                            mask_folder_name = args.mask_folder_name,
                            output_path = output_path,
                            from_scratch = args.from_scratch,
                            epochs = args.epochs,
                            weight_prefix = args.prefix,
                            batch_size = args.batch_size,
                            num_classes = args.num_classes,
                            image_size = args.image_size,
                            learning_rate = args.learning_rate,
                            optimizer = args.optimizer,
                            pretrained_weights = args.pretrained_weights,
                            model_type = args.model_type.lower()
                       )
            if args.process_type == 't':
                return
        if args.process_type == 'f':
            # trained_weights_path = '/scratch/gza5dr/Current_Canal_Experiments/Canal_Detection_Experiments/ResUNet/models/resunet_e.weights.h5'
            refined_gt_path,r_n,gt_UNRnodes,cur_UNR,common,gt_term,r_gt_term = gen_refine_gt(args.R,args.th,args.r_th,output_path,args.image_path,args.gt_path,trained_weights_path,args.model_type.lower(),i)
            args.gt_path = refined_gt_path
            args.from_scratch = False
            args.pretrained_weights = trained_weights_path
            print(args.prefix)

            with open(f'{output_path}it{i}_R{args.R}_th{args.th}_rth{args.r_th}_{args.k}dil_{args.model_type}_i{args.epochs}.csv', 'w', newline='') as csvfile:
            # Create a CSV writer object
                csvwriter = csv.writer(csvfile)

            # Write the header
                csvwriter.writerow(['it',  'Current Reachable Nodes','Current Unreachable Nodes', 'Unreachable Nodes Left from GT','Terminals Left'])  #### need to change, different csv
                csvwriter.writerow([i,r_n,cur_UNR, common,r_gt_term])




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Reachability Framework")
    parser.add_argument("--process_type", type = str, default = 'f', help = 'd = only generate data, t: only for training, r: generate refine gt, f:running the full framework')
    parser.add_argument("--iterations", type = int, default = 1, help = "number of times the process should run")
    parser.add_argument("--output_path", type = str, default = '/scratch/gza5dr/Current_Canal_Experiments/Canal_Detection_Experiments/Proposed_Model_Pipeline/implementation/framework/final_framework/output_set1_v2/', help = 'frameworks output path')
    
    ## For data generation
    parser.add_argument("--image_path", type = str, default = '/scratch/gza5dr/Canal_Datasets/NHDShape/procssed_data/full_data/processed_images/', help = 'main sattelite image folder location')
    parser.add_argument("--gt_path", type = str, default = None, help = 'path to the ground truth')
    parser.add_argument("--subfolder_prefix", type = str, default = 'main_gt', help = "any preferred name for the image and mask folder")
    parser.add_argument("--patch_size", type = int, default = 512, help = "patch size of the images and gt")
    parser.add_argument("--dilation", action='store_true', help = "if dilation is needed in the ground truth")
    parser.add_argument("--k", type = int, default = 0, help = "if dilation is true, kernel size k should be > 0")
    
    ## For training
    parser.add_argument("--model_type", type=str, default="ResUnet", help = "type of the model to use, i.e. DeepLabV3+, ResUNet, Resnet")
    parser.add_argument("--data_dir", type=str, default="/scratch/gza5dr/Canal_Datasets/NHDShape/procssed_data/",
                        help="Path to the data directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of classes for segmentation")
    parser.add_argument("--image_size", type=int, default=512, help="Size of input images")
    parser.add_argument("--img_folder_name", type=str, default="resunet_R110th0_it1output_nodialated_image_patches_512",
                        help="Name of the folder containing images, ignore if process type is 'f'")
    parser.add_argument("--mask_folder_name", type=str, default="resunet_R110th0_it1output_nodialated_mask_patches_512",
                        help="Name of the folder containing masks, ignore if process type is 'f'")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer, i.e. Adam")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--prefix", type = str, default = "resunet", help = "any preferred prefix to save the outputs") #need to modify
    parser.add_argument("--from_scratch", action='store_true', help = "if the model should run from scratch or any previous point")
    parser.add_argument("--pretrained_weights", type=str, default="",
                        help="Path to pretrained weight, should be given if the from_scratch is false")
    
    
    ## For refining GT
    parser.add_argument("--R", type=int, default=110, help = "Radius to find sources for Terminal point")
    parser.add_argument("--th", type=float, default=0.5,
                        help="threshold to use for predictions")
    parser.add_argument("--r_th", type=float, default=.01, help="threshold to connect the source and terminals")
    
    
    
    args = parser.parse_args()
    run_process(args)
    