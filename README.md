# AAaSNMD：Adversarial Attacks against Siamese Network Malware Detection

# Outline

* Dataset Generation
    * Extract Syscall
    * Syscall Category
* Target Model : Siamese Network
    * Model Analysis
    * Model Performance
* Adversarial Attack
    * Image Analysis
    * Perturbation
    * Evaluation
    * Attack Result
* Conclusion
* Future Work
* Reference

[Traditional Chinese Report @ HackMD](https://hackmd.io/@ycchiu/AAaSNMD)

# Dataset Generation
## Extract Syscall

The task at hand is to achieve adversarial attacks, with the goal of adding perturbations to malicious software in order to trick malware detectors into classifying it as benign. The previous dataset consisted of 219 samples, containing 8 families of malicious software but no benign samples. Therefore, we collected a total of 130 benign samples and extracted syscall information and corresponding time series using Cuckoo and the lab's sandbox to create the dataset for this experiment.

* As shown in the figure below, the previous data set has a total of 219 data and contains 8 categories:

```python=
mirai       :  50
unknown     :  50
penguin     :  10
fakebank    :  47
fakeinst    :  21
mobidash    :  12
berbew      :  13
wroba       :  14
```

Since we want to train a malware detector, we relabeled the above families of malware as `Malware` and labeled the benign samples as `Benign`.

* As shown in the figure below, there are a total of 349 data in the data set, and it contains 2 categories:

```python=
Benign    :  130
Malware   :  219
```

## Syscall Category

We followed the previous classification method referencing the [Searchable Linux Syscall Table for x86 and x86_64](https://filippo.io/linux-syscall-table/), dividing syscalls into 8 categories and adding a category called HighFreq. Most of the syscalls belonged to the kernel category, so we further divided them into 3 types based on their behavior and merged ipc and printk into the others category, as shown below:

```python=
syscallCategory = {
    'kernel_signal' : ['rt_sigaction', 'restart_syscall', 'kill', 'rt_sigprocmask',
                       'rt_sigsuspend','sigaltstack','tgkill','rt_sigtimedwait',
                       'sigreturn'],

    'kernel_sys' : ['uname', 'setgid', 'getpriority', 'sysinfo', 'prlimit64',
                    'geteuid', 'umask', 'getppid', 'setresuid', 'prctl', 'times',
                    'mmap', 'setsid', 'getuid', 'gettid',  'setpriority', 'setpgid',
                    'setresgid', 'getrlimit', 'getpid', 'setrlimit', 'getegid', 
                    'setuid', 'getgid', 'getpgrp', 'mmap2', 'ni_syscall'],

    'kernel_others' : ['time', 'gettimeofday', 'alarm', 'clone', 'vfork',
                       'set_tid_address', 'fork', 'set_robust_list', 'futex',
                       'nanosleep', 'wait4', 'exit', 'exit_group', 'waitpid', 
                       'ptrace', 'get_thread_area', 'set_thread_area', 
                       'clock_gettime', 'setitimer'],

    'fs' : ['stat', 'lseek', 'readlinkat', 'chroot', 'sendfile', 'umount2', 
            'symlink', 'flock', 'dup2', 'getcwd', 'chdir', 'fstat', 'mount', 
            'rmdir', 'execve', 'mkdir', 'epoll_wait', 'openat', 'eventfd2', 
            'readv', 'rename', 'epoll_create1', 'fchmod', 'pipe', 'unlink', 
            'pipe2', 'fcntl', 'open', 'read', 'write', 'lstat', 'chmod', 'readlink', 
            'getdents64', 'utimes', 'ioctl', 'select', 'access', 'close', 'poll', 
            'getdents', 'epoll_ctl', 'ftruncate', '_llseek', '_newselect', 'fcntl64',
            'fstat64', 'llseek', 'lstat64', 'renameat2', 'stat64'],

    'net' : ['accept', 'connect', 'sendto', 'shutdown', 'getsockname', 'getpeername',
             'listen', 'socketpair', 'socket', 'setsockopt', 'getsockopt', 
             'recvfrom', 'recvmsg', 'bind','recv','send','sendfile64','socketcall'],

    'mm' : ['mprotect', 'brk', 'munmap', 'madvise'],

    'sched' : ['sched_getaffinity'],
    
    'HighFreq' : ['_newselect', 'close', 'connect', 'fcntl', 'get_thread_area',
                  'getsockopt', 'open', 'read', 'recv', 'recvfrom', 
                  'rt_sigaction', 'rt_sigprocmask','sendto','socket','time'],

    'others' : [ 'getegid32', 'geteuid32', 'getgid32', 'getuid32', 'setgid32',
                 'setresuid32', 'setuid32', 'sysctl', 'ugetrlimit', 'syslog', 
                 'shmdt', 'shmget']
}

```

# Target Model : Siamese Network
## Model Analysis

The classification task of Siamese Network is to distinguish the similarity between two different inputs through the same feature extraction.

* The following table is the Ground Truth output by Siamese Network:

| Input 1  | Input 2  |  Output  |
| -------- | -------- | -------- |
| Benign   | Benign   | 1        |
| Benign   | Malware  | 0        |
| Malware  | Benign   | 0        |
| Malware  | Malware  | 1        |

## Model Performance

From the two charts below, we can see that the 19th epoch achieved the highest point in Testing Accuracy during model training, with a Testing Accuracy of 1.0 and a Training Accuracy of 0.946. By observing the subsequent trend, we also found that although Testing Accuracy and Training Accuracy occasionally fluctuated, they mostly performed well. Therefore, we chose the model from the 19th epoch as our target model.

* The figure below is the Training Accuracy and Testing Accuracy of the training process:

![](https://i.imgur.com/SSLxMHY.png)

By observing the trend of Training Loss and Testing Loss, we found that the decline is relatively consistent and there is no significant gap, which gives us enough confidence to believe that there is no Overfitting occurring.

* The figure below is the Training Loss and Testing Loss of the training process:

![](https://i.imgur.com/TP4r7eq.png)

# Adversarial Attack
## Image Analysis

After observing the results of the averaged malicious software images and comparing them to the Syscall Category, we found that none of the malicious software in the dataset had any others syscalls in any of the time periods, while benign software images had others in every time period.

* The image below shows the average of the Malware Image and the Benign Software Image:

![](https://i.imgur.com/3KVIyvR.png)

![](https://i.imgur.com/p9LDaAm.png)

## Perturbation

Based on the results of the Image Analysis, we decided to select the others syscall 'getuid32' as the perturbation to insert into the malware, as it does not affect the execution of the program. By choosing a syscall that does not affect program execution, we can preserve the functionality of the malware while ensuring that the program remains executable. Additionally, in order to maintain a relatively stable time period, we decided to insert the same number of 'getuid32' in all time periods to avoid potential changes in syscall distribution.

## Evaluation

Since the output of the Siamese Network represents the similarity between two samples, we first tested each adversarial sample against all the benign samples in the training set and calculated whether the adversarial attack was successful, i.e. whether it was similar to the benign samples, by adding up the number of successes and dividing by the total number of adversarial samples to calculate the benign_ASR (Attack Success Rate). The specific calculation can be expressed as the following formula:

$$ASR_{ben} = \cfrac {1}{N_a}\sum\limits_{i=0}^{N_{a}} [[\sum\limits_{j=0}^{N_{b}} \frac {SN(x'i, {x{b}}_{j})}{N_b} \ge 0.5 ]]$$

We also tried using the same method to test the results of the malicious samples in the training set, with the difference being that we determined whether they were "dissimilar" to the malicious samples. The specific calculation can be expressed as the following formula:

$$ASR_{mal} = \cfrac {1}{N_a}\sum\limits_{i=0}^{N_{a}} [![ \sum\limits_{j=0}^{N_{b}} \frac {SN(x'i, {x{m}}_{j})}{N_b} \lt 0.5 ]!]$$

where $SN(\cdot)$ is the Siamese Network model, ${x_{b}}$, ${x_{m}}$, and $x'$ represent benign samples in the training set, malicious samples in the training set, and adversarial samples, respectively, $N_b$, $N_m$, and $N_a$ represent the number of benign samples in the training set, the number of malicious samples in the training set, and the number of adversarial samples, respectively. $[![\cdot ]!]$ represents a conditional function that returns 1 if the condition is met, or 0 otherwise.

## Attack Result

We set the perturbation size range from 0 to 300, and observe the changes of benign_ASR and malware_ASR and draw it as the following figure:

![](https://i.imgur.com/ua4X3yA.png)

We can easily see that when no perturbation is added, the model still wrongly classifies about 11% of malicious samples as benign, which we take as the baseline for the attack. When the perturbation value is 230, the highest attack success rate of about 83.1% is achieved. At the same time, the attack success rate drops significantly when the perturbation value exceeds 230, and becomes 0% when the perturbation value exceeds 270.

We also observed that as Benign_ASR increases, Malware_ASR decreases, and vice versa. We speculate that this may be because the target is a binary classification model.

In the part where the success rate drops significantly, we speculate that it may be because the added perturbation is too large, causing the model to generate too many extreme values in the calculation process, exceeding the threshold that the model can handle, and thus affecting the activation function's judgment and causing the model to distort.

# Conclusion

Through this experiment, we have reached the following conclusions:

1. We propose an adversarial attack on a Siamese network architecture malicious software detector, which preserves the functionality of the program while considering the attack effect. The perturbation value we found can achieve the highest attack success rate of 83.1%.
2. After inserting a large perturbation value, the model will distort, leading to unreliable results in the model's judgment.

# Future Work

In this detector, it was found that the addition of perturbations to malicious samples made them more similar to benign samples and further from malicious samples. We maintain a skeptical attitude towards this result and speculate that this may be because the experiment was conducted on a binary detector. If an adversarial attack were to be conducted on a multi-class classifier, it may not be possible to easily classify a perturbed malicious sample as benign, resulting in the attack failing. In other words, perhaps we can only classify a malicious sample as another family of malicious sample, but not necessarily as a benign sample, making it difficult to bypass the malware detector. This is a direction for future research.

# Reference

* [Evading API Call Sequence Based Malware Classifiers](https://link.springer.com/content/pdf/10.1007/978-3-030-41579-2_2.pdf)
