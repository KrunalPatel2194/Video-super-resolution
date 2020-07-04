import os, subprocess, sys, datetime, signal, shutil

runcase = int(sys.argv[1])
print ("Testing test case %d" % runcase)

def preexec(): # Don't forward signals.
    os.setpgrp()
    
def mycall(cmd, block=False):
    if not block:
        return subprocess.Popen(cmd)
    else:
        return subprocess.Popen(cmd, preexec_fn = preexec)
    
def folder_check(path):
    try_num = 1
    oripath = path[:-1] if path.endswith('/') else path
    while os.path.exists(path):
        print("Delete existing folder " + path + "?(Y/N)")
        decision = input()
        if decision == "Y":
            shutil.rmtree(path, ignore_errors=True)
            break
        else:
            path = oripath + "_%d/"%try_num
            try_num += 1
            print(path)
    
    return path


if( runcase == 1 ): # inference a trained model
    
    dirstr = './results/' # the place to save the results
    testpre = ['city'] # the test cases

    if (not os.path.exists(dirstr)): os.mkdir(dirstr)
    
    # run these test cases one by one:
    for nn in range(len(testpre)):
        cmd1 = ["python", "main.py",
            "--cudaID", "0",            # set the cudaID here to use only one GPU
            "--output_dir",  dirstr,    # Set the place to put the results.
            "--summary_dir", os.path.join(dirstr, 'log/'), # Set the place to put the log. 
            "--mode","inference", 
            "--input_dir_LR", os.path.join("./LR/", testpre[nn]),   # the LR directory
            #"--input_dir_HR", os.path.join("./HR/", testpre[nn]),  # the HR directory
            # one of (input_dir_HR,input_dir_LR) should be given
            "--output_pre", testpre[nn], # the subfolder to save current scene, optional
            "--num_resblock", "16",  # our model has 16 residual blocks, 
            "--checkpoint", './model/model',  # the path of the trained model,
            "--output_ext", "png"               # png is more accurate, jpg is smaller
        ]
        mycall(cmd1).communicate()

    
elif( runcase == 2): # Train the model
    
    #Pre-trained VGG19   
    VGGPath = "model/" # the path for the VGG model, there should be a vgg_19.ckpt inside
    VGGModelPath = os.path.join(VGGPath, "vgg_19.ckpt")
    #Pretraied FRVSR model
    
    FRVSRModel = "model/FRVSR" 
    #put training data in below folder
    TrainingDataPath = "TrainingDataPath" 
    
    '''Prepare Training Folder'''
    # path appendix, manually define it, or use the current datetime, now_str = "mm-dd-hh"
    now_str = datetime.datetime.now().strftime("%m-%d-%H")
    train_dir = folder_check("model%s/"%now_str)
    cmd1 = ["python", "main.py",
        "--cudaID", "0", # set the cudaID here to use only one GPU
        "--output_dir", train_dir, # Set the place to save the models.
        "--summary_dir", os.path.join(train_dir,"log/"), # Set the place to save the log. 
        "--mode","train",
        "--batch_size", "4" , # small, because GPU memory is not big
        "--RNN_N", "10" , # train with a sequence of RNN_N frames, >6 is better, >10 is not necessary
        "--movingFirstFrame", # a data augmentation
        "--random_crop",
        "--crop_size", "32",
        "--learning_rate", "0.00005",
        # -- learning_rate step decay, here it is not used --
        "--decay_step", "500000", 
        "--decay_rate", "1.0", # 1.0 means no decay
        "--stair",
        "--beta", "0.9", # ADAM training parameter beta
        "--max_iter", "500000", # 500k or more, 
        "--save_freq", "10000", # the frequency we save models
        # -- network architecture parameters --
        "--num_resblock", "16", 
        # -- VGG loss, disable with vgg_scaling < 0
        "--vgg_scaling", "0.2",
        "--vgg_ckpt", VGGModelPath, # necessary if vgg_scaling > 0
    ]
    
    cmd1 += [
        "--input_video_dir", TrainingDataPath, 
        "--input_video_pre", "scene",
        "--str_dir", "2000",
        "--end_dir", "2250",
        "--end_dir_val", "2290",
        "--max_frm", "119",
        # -- cpu memory for data loading --
        "--queue_thread", "12",# Cpu threads for the data. >4 to speedup the training
        "--name_video_queue_capacity", "1024",
        "--video_queue_capacity", "1024",
    ]
    
    
    cmd1 += [ # based on a pre-trained FRVSR model. Here we want to train a new adversarial training
        "--pre_trained_model", # True
        "--checkpoint", FRVSRModel,
    ]
    
    
    
    ''' parameters for GAN training '''
    cmd1 += [
        "--ratio", "0.01",  # the ratio for the adversarial loss from the Discriminator to the Generator
        "--Dt_mergeDs",     # if Dt_mergeDs == False, only use temporal inputs, so we have a temporal Discriminator
                            # else, use both temporal and spatial inputs, then we have a Dst, the spatial and temporal Discriminator
    ]
    
    cmd1 += [ # here, the fading in is disabled 
        "--Dt_ratio_max", "1.0",
        "--Dt_ratio_0", "1.0", 
        "--Dt_ratio_add", "0.0", 
    ]
    ''' Other Losses '''
    cmd1 += [
        "--pingpang",           # our Ping-Pang loss
        "--pp_scaling", "0.5",  # the weight of the our bi-directional loss, 0.0~0.5
        "--D_LAYERLOSS",        # use feature layer losses from the discriminator
    ]
    
    pid = mycall(cmd1, block=True) 
    try: # catch interruption for training
        pid.communicate()
    except KeyboardInterrupt: # Ctrl + C to stop current training try to save the last model 
        print("runGAN.py: sending SIGINT signal to the sub process...")
        pid.send_signal(signal.SIGINT)
        # try to save the last model 
        pid.communicate()
        print("runGAN.py: finished...")
        
        