import numpy as np
import torch
import siamese_net
import mlflow

## Load Siamese Network
model = siamese_net.SiameseNetwork()

## Load  Model Parameter
model.load_state_dict(torch.load("./model/SN_Detector.pt"))
model.eval()

## Load Dataset
imgs = np.load("./Data/all_img.npy")
labels = np.load("./Data/all_labels.npy", allow_pickle = True)
max_data = 2110
test_imgs = []
for i in range(len(labels)):
    if labels[i] != "benign":
        test_imgs.append(imgs[i])

## Load Training Data
train_data = np.load("./Traindata/train_data.npy")
train_labels = np.load("./Traindata/train_labels.npy", allow_pickle = True)
train_data_mal = []
train_data_ben = []
for i in range(len(train_labels)):
    if train_labels[i] == "benign":
        train_data_ben.append(train_data[i] / max_data)
    else:
        train_data_mal.append(train_data[i] / max_data)

mlflow.set_experiment("Adversarial_Attacks")
mlflow.start_run(run_name = 'Perturbation Test')

for perturbation in range(0, 301, 10):
    test_samples = []
    for test_img in test_imgs:
        for i in range(len(test_img)):
            test_img[i][-1] = perturbation
        test_samples.append(test_img / max_data)

    ben_test_result = []
    ben_ASR = 0
    for test_sample in test_samples:
        test_sample = torch.from_numpy(np.array([test_sample])).to(torch.float).unsqueeze(1).clone()
        tmp = []
        for ben_img in train_data_ben:
            ben_img = torch.from_numpy(np.array([ben_img])).to(torch.float).unsqueeze(1).clone()
            tmp += [round(model(ben_img, test_sample).item())]
        ben_ASR += round(sum(tmp) / len(train_data_ben))
        ben_test_result.append(tmp)
    ben_ASR /=  len(test_samples)
    print('Ben ASR :', ben_ASR * 100, '%')

    mlflow.log_metric("benign_ASR", ben_ASR, step = perturbation)

    mal_test_result = []
    mal_ASR = 0
    for test_sample in test_samples:
        test_sample = torch.from_numpy(np.array([test_sample])).to(torch.float).unsqueeze(1).clone()
        tmp = []
        for mal_img in train_data_mal:
            mal_img = torch.from_numpy(np.array([mal_img])).to(torch.float).unsqueeze(1).clone()
            tmp += [round(model(mal_img, test_sample).item())]
        mal_test_result.append(tmp)
        mal_ASR += round(sum(tmp) / len(train_data_mal))
    mal_ASR /= len(test_samples)
    print('Mal ASR :', (1 - mal_ASR) * 100, '%')

    mlflow.log_metric("malware_ASR", mal_ASR, step = perturbation)
mlflow.end_run()