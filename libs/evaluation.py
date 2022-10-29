import torch

# SR : Segmentation Result
# GT : Ground Truth

def dice_coeff_ml(SR, GT):
    smooth = 1.
    num = SR.size(0)
    m1 = SR.view(num, -1)  # Flatten
    m2 = GT.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
 
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def get_accuracy_ml(SR,GT,threshold=0.5):
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)
    acc = float(corr)/float(tensor_size)

    return acc

def get_sensitivity_ml(SR,GT,threshold=0.5):
    # Sensitivity == Recall

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1).long()+(GT==1).long()) == 2
    FN = ((SR==0).long()+(GT==1).long()) == 2

    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)     
    
    return SE

def get_specificity_ml(SR,GT,threshold=0.5):

    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0).long()+(GT==0).long())==2
    FP = ((SR==1).long()+(GT==0).long())==2



    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP

def get_precision_ml(SR,GT,threshold=0.5):
    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1).long()+(GT==1).long())==2
    FP = ((SR==1).long()+(GT==0).long())==2

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC

def get_F1_ml(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity_ml(SR,GT,threshold=threshold)
    PC = get_precision_ml(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS_ml(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)
    
    Inter = torch.sum((SR.long()+GT.long())==2)
    Union = torch.sum((SR.long()+GT.long())>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC_ml(SR,GT,threshold=0.5):
    # DC : Dice Coefficient

    Inter = torch.sum((SR.long()+GT.long())==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC

####################################

def dice_coeff(SR, GT):
    smooth = 1.
    num = SR.size(0)
    m1 = SR.view(num, -1)  # Flatten
    m2 = GT.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
 
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1).long()+(GT==1).long()) == 2
    FN = ((SR==0).long()+(GT==1).long()) == 2

    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)     
    
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0).long()+(GT==0).long())==2
    FP = ((SR==1).long()+(GT==0).long())==2



    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1).long()+(GT==1).long())==2
    FP = ((SR==1).long()+(GT==0).long())==2

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)
    
    Inter = torch.sum((SR.long()+GT.long())==2)
    Union = torch.sum((SR.long()+GT.long())>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR.long()+GT.long())==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC



