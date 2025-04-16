from imports import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
master_iter = 0

norm_fn = torchvision.transforms.Normalize(mean=[77.52425988, 77.52425988, 77.52425988],std=[51.8555656, 51.8555656, 51.8555656])
                                       

def train_model_loss_attenuation(args, INPUT_CHK_PT, THRESHOLD, exp_name):
    '''
    training with learned loss attenuation for aleatoric uncertainty estimation
    '''
    # Start training
    print('                                                                       ')
    print("Model initialization for training with learned loss attenuation for aleatoric uncertainty estimation ...", flush=True)

    aucs, aucs_training, losses, sensitivities, specificities = [], [], [], [], []
    
    
    if args.random_state is not None:
        torch.manual_seed(args.random_state)
    
    # # loading model
    model_dict = dict(
        type='ImageClassifier',
        backbone=dict(
            type='ConvNeXt',
            arch='small',
            out_indices=(3, ),
            drop_path_rate=0.4,
            gap_before_final_norm=True,
            init_cfg=[
                dict(
                    type='TruncNormal',
                    layer=['Conv2d', 'Linear'],
                    std=0.02,
                    bias=0.0),
                dict(type='Constant', layer=['LayerNorm'], val=1.0, bias=0.0)
            ]),
            head=dict(
                type='LinearClsHead',
                num_classes=2,
                in_channels=768,
                loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))

    
    
    model = MODELS.build(model_dict).to('cuda')
    # model.cuda(4)
    print("Loading checkpoint ...", flush=True)
    load_checkpoint(model, INPUT_CHK_PT)

    # This code modifies the output layer of a model by replacing the last fully connected layer (module.fc) 
    # in the model's head with a new one that has two output features (instead of the original number of features)
    # By specifying 2 output features, the model now has an additional output. It represents variance because the goal is to output 
    # both a value (mean prediction) and its uncertainty (variance) in tasks like probabilistic modeling or uncertainty estimation.

    # this piece of code (674-676) is only for loss attenuation
    for key, module in model.named_children():
        if key == 'head':
            module.fc = nn.Linear(768, 2)

    # # apply dropout if required
    if args.dropout_rate is not None:
        append_dropout(model, args.dropout_rate)
        #print(model)

    # # freeze first number of layers if specified
    if args.fine_tuning == 'partial':
        model = apply_custom_transfer_learning(model, args.upto_freeze)

    model.to('cuda')
    
    # # load training and validation dataset
    print("Loading data ...", flush=True)
    train_dataset = load_data_from_csv(args, args.train_csv_pth, training_flag=True)
    valid_dataset = load_data_from_csv(args, args.valid_csv_pth, training_flag=False)
    valid_index = list(range(len(valid_dataset)))
    train_index = list(range(len(train_dataset)))
    
    # get the labels and dataset types out
    train_labels = [train_dataset.labels[i] for i in train_index]
    train_datasettypes = [train_dataset.datasettype[i] for i in train_index]
    valid_labels = [valid_dataset.labels[i] for i in valid_index]
    valid_datasettypes = [valid_dataset.datasettype[i] for i in valid_index]

    if args.sampler == 'Weighted':
        print('Using weighted sampler to generate training loader')
        sample_weights = compute_class_weights(train_labels)
        weighted_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=weighted_sampler,num_workers=args.threads, pin_memory=True, prefetch_factor=args.prefetch_factor)
        
    elif args.sampler == 'Balanced':
        print('Using balanced sampler to generate training loader')
        balanced_sampler = BalancedBatchSampler(train_labels, args.batch_size)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=balanced_sampler, num_workers=args.threads, pin_memory=True, prefetch_factor=args.prefetch_factor)

    elif args.sampler == 'Simple':
        print('Creating loaders without balancing or weighing')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.threads, pin_memory=True, prefetch_factor=args.prefetch_factor)
    
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads,
                              pin_memory=True, prefetch_factor=args.prefetch_factor)
    
    # Plotting and counting images per batch
    #count_and_plot_batch_data(train_loader, args.out_dir)
     

    # Count cancer +ve and cancer -ve training set
    pos_count_train = sum(label == 1 for label in train_labels)
    neg_count_train = sum(label == 0 for label in train_labels)
    print(f"-------------------------------------------------------------------------------------------------------------")
    print(f"Training set: {pos_count_train} cancer-positive, {neg_count_train} cancer-negative", flush=True)

    # Count cancer +ve and cancer -ve validation set
    pos_count_valid = sum(label == 1 for label in valid_labels)
    neg_count_valid = sum(label == 0 for label in valid_labels)
    print(f"-------------------------------------------------------------------------------------------------------------")
    print(f"Validation set: {pos_count_valid} cancer-positive, {neg_count_valid} cancer-negative", flush=True)
    print(f"-------------------------------------------------------------------------------------------------------------")

    # Loss function
    if args.loss_function == 'FocalLoss':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    elif args.loss_function == 'BCELoss':
        criterion = nn.BCELoss(reduction='none')
    elif args.loss_function == 'BCELogitsLoss':
        criterion = nn.BCEWithLogitsLoss(reduction='none') 
    elif args.loss_function == 'Softmax':
        criterion = SoftmaxEQLoss(num_classes=2)
   
    # # select the optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.start_learning_rate, betas=(0.9, 0.999), weight_decay=0.0)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.start_learning_rate, momentum=0, weight_decay=0.01)
    elif args.optimizer == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.start_learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    else:
        print('ERROR. UNKNOWN optimizer.')
        return

    # # learning rate scheduler
    my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_every_N_epoch, gamma=args.decay_multiplier)
   
    print("Start epoch ...", flush=True)
    
    writer = []
    for epoch in range(args.num_epochs):
        # # train for one epoch
        start_time = datetime.datetime.now().replace(microsecond=0)
        
        # Training call
        avg_loss, auc_per_epoch = run_train_loss_attenuation(train_loader, model, criterion, optimizer, my_lr_scheduler, writer, args.t_number)
        losses.append(avg_loss)
        aucs_training.append(auc_per_epoch)

        my_lr_scheduler.step()
        my_lr = my_lr_scheduler.get_last_lr()
        
        
        if epoch % args.save_every_N_epochs == 0 or epoch == args.num_epochs-1:
            
            # validation call
            (auc_val, specificity, sensitivity) = run_validate_loss_attenuation(valid_loader, model, args, writer)
            
            end_time = datetime.datetime.now().replace(microsecond=0)
            time_diff = end_time - start_time

            aucs.append(auc_val)
            sensitivities.append(sensitivity)
            specificities.append(specificity)

            print(f"> Epoch:{epoch + 1} Training Loss:{round(avg_loss, 4)} AUC_Training:{round(auc_per_epoch, 4)} AUC_Tuning:{round(auc_val, 4)} Sensitivity:{round(sensitivity, 4)}Specificity:{round(specificity, 4)} Time Taken {epoch+1}: {time_diff}", flush=True)
            #save_checkpoint({'epoch': epoch + 1,'arch': 'ConvNeXt','state_dict': model.state_dict(),'auc': auc_val,'optimizer': optimizer.state_dict(),}, os.path.join(args.out_dir, 'checkpoint__' + str(epoch) + '.pth.tar'))
    
    # # save the last epoch model for deployment
    last_model_path = os.path.join(args.out_dir, 'last_epoch_model.pth')
    save_checkpoint({'arch': 'ConvNeXt','state_dict': model.state_dict(),'optimizer': optimizer.state_dict(),}, last_model_path)

    epochs = list(range(1, args.num_epochs + 1))  

    # Plot training losses and aucs
    plot_training_metrics(epochs, losses, aucs_training, args.out_dir, args.exp_name)

    # Plot Sensitivity and Specificity
    plot_sensitivity_specificity_auc(epochs, sensitivities, specificities, aucs, args.out_dir, args.exp_name)

def run_train_loss_attenuation(train_loader, model, criterion, optimizer,  my_lr_scheduler, writer, t_num):
    """ Function that runs the training with learned loss attenuation
    """
    global master_iter

    # switch to train mode
    model.train()
    avg_loss = 0
    # new
    all_labels, all_preds = [], []
    
    for i, (images, target, datasettype, index) in enumerate(train_loader):
        # # measure data loading time
        master_iter += 1
        
        images = images.cuda()
        images = norm_fn(images.float())
        target = target.cuda()
       
        optimizer.zero_grad()
        output = model(images)
        mu, sigma = output.split(1, 1)

        # # monte carlo sampling for learned loss attenuation
        loss_total = torch.zeros(t_num, target.size(0))
        
        for t in range(t_num):
            # assume that each logit value is drawn from Gaussian distribution, 
            # therefore the whole logit vector is drawn from multi-dimensional Gaussian distribution
            epsilon = torch.randn(sigma.size()).cuda()
            logit = mu + torch.mul(sigma.pow(2), epsilon)
            # # compute loss for each monte carlo sample
            loss_total[t] = criterion(torch.sigmoid(torch.flatten(logit)), target.float())
        
        # # compute average loss
        sample_loss = torch.mean(loss_total, 0)
        loss = torch.mean(sample_loss)
        avg_loss += loss.item()
        
        # # compute gradient 
        loss.backward()
        optimizer.step()

        # new
        all_labels.extend(target.cpu().numpy())
        all_preds.extend(torch.sigmoid(logit).cpu().detach().numpy())
    
    # Calculate AUC at the end of the epoch
    if len(all_labels) > 0 and len(all_preds) > 0:
        auc = roc_auc_score(all_labels, all_preds)
    else:
        auc = 0.0  # Handle case with no data

    return avg_loss/len(train_loader), auc

def run_validate_loss_attenuation(val_loader, model, args, writer):

    global master_iter

    # # switch to evaluate mode
    model.eval()
    
    type_all,predicted_labels,logits_all,scores_all,sigma_all,datasettype_all  = [],[],[],[],[],[]
    total_correct = 0
    total_samples = 0  # to count the total number of samples for averaging

    # Subset data
    subset_metrics = {
        'csaw_canpos': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'total': 0},
        'csaw_canneg': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'total': 0},
        'embed_canpos': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'total': 0},
        'embed_canneg': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'total': 0}
    }

    with torch.no_grad():
        for i, (images, target, datasettype, index) in enumerate(val_loader):
            # # compute output
            images = images.cuda()

            images = norm_fn(images.float())
           
            output = model(images)
            test_mu, test_sigma = output.split(1, 1)
            # test_mu the mean (prediction) of the model.
            # test_sigma is standard deviation or uncertainty of the prediction
            # sigma refers to the uncertainty (or standard deviation) associated 
            # with the model's predictions. It's used to quantify how confident the model is in its output. 
            output = test_mu
            test_sigma = F.softplus(test_sigma) # added by me
            # Softplus ensures that all values in test_sigma are positive
            # makes it easy for representing std dev or variances
            # used when modeling a probabilistic output where test_sigma should represent 
            # actual standard deviations or variances.
            sigma_out = torch.flatten(test_sigma)
            sigma = list(sigma_out.cpu().numpy())
            sigma_all += sigma

            # Make predictions
            target_image_pred_logits = torch.flatten(output)
            target_image_pred_probs = torch.sigmoid(target_image_pred_logits)
            
            # Convert probabilities to binary labels using threshold
            predicted_label = (target_image_pred_probs > THRESHOLD).long() # on CUDA
           
            # # accumulate 
            labl_list = list(target.cpu().numpy()) # true labels
            type_all += labl_list
           
            logit = list(target_image_pred_logits.cpu().numpy())
            logits_all += logit

            scr = list(target_image_pred_probs.cpu().numpy())
            scores_all += scr
            # make changes to account for plots for diseaseed and non diseased

            correct_predictions = torch.sum(predicted_label == target.float().cuda()).item()  # Ensure target is on CUDA

            total_correct += correct_predictions

            total_samples += target.size(0)  # Count the number of samples

            predicted_labels += list(predicted_label.cpu().numpy())

            datasettype_all += list(datasettype)

            target = target.cpu()

            # Accumulate subset metrics based on dataset type and labels
            for img, label, dtype, p_label in zip(images, target, datasettype, predicted_labels):
                if dtype == 'csaw':
                    if label == 1:
                        subset_metrics['csaw_canpos']['total'] += 1
                        if p_label == 1:
                            subset_metrics['csaw_canpos']['tp'] += 1
                        else:
                            subset_metrics['csaw_canpos']['fn'] += 1
                    elif label == 0:
                        subset_metrics['csaw_canneg']['total'] += 1
                        if p_label == 0:
                            subset_metrics['csaw_canneg']['tn'] += 1
                        else:
                            subset_metrics['csaw_canneg']['fp'] += 1
                elif dtype == 'embed':
                    if label == 1:
                        subset_metrics['embed_canpos']['total'] += 1
                        if p_label == 1:
                            subset_metrics['embed_canpos']['tp'] += 1
                        else:
                            subset_metrics['embed_canpos']['fn'] += 1
                    elif label == 0:
                        subset_metrics['embed_canneg']['total'] += 1
                        if p_label == 0:
                            subset_metrics['embed_canneg']['tn'] += 1
                        else:
                            subset_metrics['embed_canneg']['fp'] += 1


    # save bunch of csv files
    result_df1 = pd.DataFrame(list(zip(type_all, predicted_labels, logits_all, scores_all, sigma_all, datasettype_all)), columns=['true_labels', 'predicted_labels', 'logits', 'score', 'uncertainty', 'dataset'])
    csaw_df = result_df1[result_df1['dataset'] == 'csaw']
    embed_df = result_df1[result_df1['dataset'] == 'embed']
    
    if args.bsave_valid_results_at_epochs:
        results_path1 = os.path.join(args.out_dir, 'results_' + args.exp_name + str(master_iter+1) + '.tsv')
        result_df1.to_csv(results_path1, sep='\t', index=False)
    
    results_path2 = os.path.join(args.out_dir, f'results_last_{args.exp_name}.tsv') 
    result_df1.to_csv(results_path2, sep='\t', index=False)

    #  calc AUC from ROC
    fpr, tpr, _ = roc_curve(np.array(type_all), np.array(scores_all), pos_label=1) # type_all==true labels and scores_all = prob of predicted scores
    fpr_csaw, tpr_csaw, _ = roc_curve(csaw_df['true_labels'], csaw_df['score'], pos_label=1)
    fpr_embed, tpr_embed, _ = roc_curve(embed_df['true_labels'], embed_df['score'], pos_label=1)
    # tpr is sensitivity
    
       
    auc_val = auc(fpr,tpr)
    auc_csaw = auc(fpr_csaw, tpr_csaw)
    auc_embed = auc(fpr_embed, tpr_embed)

    true_labels = np.array(type_all)
    pred_labels = np.array(predicted_labels)
    
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = conf_matrix.ravel()

    specificity = tn / (tn + fp)
    sensitivity = tp/ (tp + fn)

    # Log file
    log_file = os.path.join(args.out_dir, 'log.log')

    # Check if the file exists to decide whether to write the header
    if not os.path.exists(log_file):
        with open(log_file, 'w') as fp:
            # Write the header line
            fp.write("Master Iter\tAUC\tSpecificity\tSensitivity\n")

    # Append the data
    with open(log_file, 'a') as fp:
        fp.write("{:d}\t{:1.5f}\t{:1.5f}\t{:1.5f}\n".format(master_iter, auc_val, specificity, sensitivity))



    # Plot logits scores
    # plot_logits_scatter(result_df1, args.out_dir, args.exp_name)
    plot_logits_distribution(logits_all, args.out_dir, args.exp_name)
    plot_logits_distribution_by_confusion_matrix(logits_all, type_all, args.out_dir, args.exp_name, THRESHOLD)
    plot_logits_distribution_by_dataset(logits_all, datasettype_all, args.out_dir, args.exp_name)
    plot_logits_distribution_by_dataset2(result_df1, args.out_dir, args.exp_name)

    # Plot ROC curves
    plot_roc_curve(fpr,tpr,auc_val,fpr_csaw,tpr_csaw,auc_csaw,fpr_embed,tpr_embed,auc_embed,args.out_dir,args.exp_name)
    #plot_csaw_roc_curve(fpr,tpr,auc_val,args.out_dir,args.exp_name)
    
    # Plot score distribution overall 
    plot_score_distribution(result_df1, args.out_dir, args.exp_name)

    # Plot score distribution for csaw and embed
    plot_score_distribution_datasets(result_df1, args.out_dir, args.exp_name)

    # Plot uncertainty distribution
    #This plot will show the distribution of uncertainty values for each predicted label (0 and 1).
    # Higher peaks will indicate more common uncertainty values for that label.
    plot_uncertainty_distribution(result_df1, args.out_dir, args.exp_name)

    # Save metrics from the result file i.e last epoch result tsv
    save_last_epoch_metrics(args.out_dir, args.exp_name)

    return (auc_val, specificity, sensitivity) 


