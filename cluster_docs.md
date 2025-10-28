
---
### Cluster Quick Start Guide

The CS HPC cluster is intended for research use only.  For this reason MSc and undergraduate students are only granted access when working on a research project, and should first have their code working on the departmental timeshare machines (GPU: blaze, non-GPU: short).

### Cluster Accounts

Please fill in [this](https://hpc.cs.ucl.ac.uk/account-form/) form in order to get an account on the CS cluster.  You will need a CS account before you can get a cluster account.  Everyone in the CS department already has one of these, if you need to apply for one there is an option on the same form. Your cluster account credentials are the same as your CS account credentials.

If you don’t know your CS account credentials you will need to pick these up in person.  The account details can be collected from room 4.20 Malet Place Engineering Building.  Please bring your UCL ID card with you as proof of ID.

### Accessing the Cluster

To access the cluster  you need to ssh into a login node. In order to be able to ssh to a login node, you first need to be on the Computer Science network. To do this, first log into one of the CS gateway machines. These are tails.cs.ucl.ac.uk (for staff) and knuckles.cs.ucl.ac.uk (for students, including MSc).

First, ssh to the gateway node:

`ssh username@knuckles.cs.ucl.ac.uk`

replacing username with your CS username.

From here, ssh to the appropriate login node:

- GPU Service – beaker.cs.ucl.ac.uk
- CMIC/MedPys – comic2.cs.ucl.ac.uk
- Biosciences, SLMS, UCLEB & BLIC – pchuckle.cs.ucl.ac.uk
- City of London Cancer Centre – gamble.cs.ucl.ac.uk
- Economics/Stats/Financial Computing – peacock.cs.ucl.ac.uk
- Computer Science – vic.cs.ucl.ac.uk
- Computer Sciences – NLP/MR/CSML – pryor.cs.ucl.ac.uk
- Computer Science Bioinfomatics – morecambe2.cs.ucl.ac.uk
- Other – wise.cs.ucl.ac.uk (currently still on old cluster, but moving soon)

If for any reason your assigned node is not working you can use one of the others.

For further information about ssh-ing, including how to set up an ssh tunnel, [click here](https://hpc.cs.ucl.ac.uk/ssh-scp/).

### Making Best Use of the Cluster

You will need to be familiar with Linux in order to use the cluster.  If you need to brush up on your skills then [software-carpentry.org](https://software-carpentry.org/) offer a lesson on the [unix shell.](http://swcarpentry.github.io/shell-novice/)  You should work through this along with the [shell-extras](https://carpentries-incubator.github.io/shell-extras/) lesson.  This will take you through all the concepts that you will need to know in order to use the cluster.  We offer an hour long cluster induction session where we will take you through the services that are available and their proper usage.  These are run on demand, if you would like to book one then please [contact us](https://hpc.cs.ucl.ac.uk/contact-us/).

Please see the [full guide](https://hpc.cs.ucl.ac.uk/full-guide/) for additional information.

---
#### HPC using Grid Engine Anatomy of a Job Script 
1. # This is an example of a job script. 
2. #$ -S /bin/bash 
3. #$ -l h_rt=0:10:0 
4. #$ -l h_vmem=2G,tmem=2G 
5. #$ -wd /home/users/dg/work 
6. #$ -j y 
7. #$ -N NCBIAllGenomes 
8. # Commands to be executed go here… 
9. /home/users/dg/SGEtest/md_single

HPC using Grid Engine Anatomy of a Job Script 
1. #This is a simple example of a job script.  This is a comment (begins with a #). 
2. #$ -S /bin/bash  The first GE command. Note that the line begins with #$. It is NOT a comment, because of the $ sign and requires the # sign, so do not remove it. The string following is a resource request. -S specifies the shell that the job will run under, followed by the path to Bash. 
3. #$ -l h_rt=0:10:0  Request the run-time for this job (in hours:minutes:seconds). Note: if you request a run-time that is insufficient for your job to complete, GE will kill your job when it has had its allotted time, with no warning (job no longer appears in the queue, premature output termination, no errors). Large over-provision of run-time in a request, may increase queuing time.
4. #$ -l h_vmem=2G,tmem=2G  This is a memory request and is required (otherwise the job will not run and simply stay in the queue). Note that there are two options in this request; h_vmem and tmem, and their values are given in gigabytes (G). Both these options must be given in a memory request and they should be the same value. The technical reasons why memory is requested this way are beyond the scope of this course, but they allow GE to properly manage the memory of the job and keep track of memory usage on the compute node. Requesting memory for a job is somewhat of an art. Requesting too much memory wastes resources and increases queuing time, whereas requesting too little causes the job process to swap, significantly increasing its run-time (which could mean it does not complete). It is important to be aware of an application’s memory usage, so an appropriate memory request can be made. Requesting memory rule of thumb: Benchmark your job! You can do this by requesting an interactive session on a compute node using the command “qrsh –l h_vmem=nG,tmem=nG” where n is the amount of memory you wish to request, then running your application in the background. To do this, add an & symbol to the end of the command line. Once the application is running, use the “top” command to monitor the system resources , including memory, your application is using. It is only necessary to run the application in the background long enough to obtain a roughly consistent memory usage value. Use this value plus 0.5GB for the memory request in your job script.
5.  #$ -wd /home/users/dg/work  This is the directory from where the job script was submitted. 
6. #$ -j y  This request means “join yes”. When a job is submitted, GE writes two status files to the home directory by default; a .o and a .e. The .o file contains stdout output which is generally anything that would normally be written to the screen if the application was run interactively. The .e file contains stderr output. It is normal, on job completion, for the .e file to be empty and as some error output may be written to the .o file anyway, to save generating a pair of files for each job, the .e file can be “joined” to the .o file. This simply means any output normally written to the .e file, will instead be written to the .o file and no .e file will be created. 
7. #$ -N NCBIAllGenomes  Name the job with the following string. If this request is not made, the job will take the name of the job script. The job name has an associated environment variable, JOB_NAME, which can be used to reference the job in a script (the name can also be used to hold other jobs from running until completion).
8. # Commands to be executed go here…  Another comment. 
9. /home/users/dg/SGEtest/md_single  The first normal shell command, usually the path to the application plus options, input and output directives, as would be given on the command line. As a job script is fundamentally a shell script, this could be additional shell commands, for example to preprocess data before the application is called. The shell commands in a job script should be written exactly as if the command was to be run interactively. It is a good idea to include the “hostname” and “date” commands as the first commands in the job script, so the node accepting the job and the start time can be determined. This is useful if the job errors in a way that suggests an issue with the node or the cluster.

Once a job script has been created, it can be submitted to GE for processing. There are three principal commands for managing jobs in GE: 
• qsub – submit a job script to the queue. The GE options in the job script can also be given on the qsub command line. There is an associated environment variable, JOB_ID, which can be used to reference the job: dg@nexus:~/SGEtest$ qsub AllGenomeDownload.sh Your job 25 ("NCBIAllGenomes") has been submitted 
• qdel – delete a job from the queue. A job ID is required: dg@nexus:~/SGEtest$ qdel 25<.${SGE_TASK_ID}(- array job)> dg has deleted job 25 
• qstat – check the status of a submitted job.

dg@nexus:~/SGEtest$ qstat job-ID prior name user state submit/start at queue slots ja-task-ID ----------------------------------------------------------------------------------------------------------------- 25 0.55500 NCBIAllGen dg r 04/23/2018 13:12:44 all.q@fry-611-48.local 1 dg@nexus:~/SGEtest$ 
Fields to note: 
• job-ID  from qsub command. 
• name  from the –N option, if given in the job script. Otherwise, the job script name. 
• slot  number of cores the job is using (normally, no. of slots = no. of cores = no. of threads). 
• state  status of the job in the queue. There are four possible states: qw, r, Eqw and s.
• qw – “queue wait”. The normal state of a job that has just been submitted. The job is waiting for its requested resources to become available. 
• r – “running”. The job is running! 
• Eqw – “Error queue wait”. When a job is first submitted, GE evaluates the script to make sure there are no issues. If there are none, the job remains in qw state, waiting to run. If, however, an issue is found, the jobs goes into Eqw state. In this state, the job will remain in the queue, but it will not run until the error is corrected and the job resubmitted. The cause of the error can be found with the command “qstat –f –j ” This gives a much more comprehensive description about the submitted job. Look for the string “error reason” in the output, to see what GE considers the issue to be. 
• s – “Suspended”. If a job is in “s” state, it has been suspended. Jobs are suspended when they are causing issues on the cluster e.g. heavy I/O. During a suspension, no attempt should be made to restart the job, for example, by deleting and resubmitting it. Once the nature of the issue has been determined, the job will be resumed.
• #$ -o|e /alternative/directory/path  Use the –o and/or –e flag to redirect the .o and/or .e files to an alternative output directory (the default is the home directory, unless the -cwd or –wd flag was given). 
• #$ -cwd  This is the directory from where the job script was submitted. 
• #$ -V  Import the users .bashrc environment into the job.
• #$ -t 1-10[:2]  Submits a so-called array job. An array job consists of a series of tasks. The tasks are scheduled independently and can run concurrently, very much like separate jobs. An array job can be accessed as a single unit or as individual tasks, by qdel for example. 
• The option argument to -t specifies the number of array job tasks and the index number which will be associated with each task. The index numbers will be exported to the job tasks via the environment variable SGE_TASK_ID. The task id range specified may be a single number or a simple range. The :2 is the task step size. If applied here, only tasks 1,3,5,7,9 would run. When absent, all tasks in the specified range are run. The task id is shown in the “ja-task-ID” output from qstat. 
• #$ -tc 5  Define the maximum number of concurrently running tasks for the array job. In this case, only the first 5 tasks would be run [or the first 5 tasks for the given step size]. 
• The STDOUT and STDERR of array job tasks will be written into different files with the default name: .e|o.. In order to change this default, the -e and -o options can be used together with the environment variables $JOB_NAME, $JOB_ID and $SGE_TASK_ID.
• $SGE_TASK_ID can also be used to refer to lines in a file if, for example, there is a data-point on each line.This can be done in a few different ways, for example: 
• sed -n ${SGE_TASK_ID}'{p;q}' input.data  The -n flag suppresses the ‘normal’ output of sed with p, which is to print the file given. The q quits sed as soon as the specified line is reached, which minimises unnecessary processing of the input file. 
• head -${SGE_TASK_ID} input.data | tail -1 head -[n] prints the first N lines. tail -[n] prints the last N lines. In this case, tail -1 is being run on a subset of the initial file (output of the head command), which has the Nth line as the final entry.
• #$ -pe smp 8  “[Request a] parallel environment, symmetric multi-processing [with] 8 [cores]”. Use this environment to run multi-threaded or OpenMP applications over multiple cores within a compute node. You must also include #$ -R y in your job script, to reserve the cores requested. 
• In a parallel environment, the values of tmem and h_vmem are multiplied by the number of cores requested. This is because the scheduler interprets these values as being ‘per thread’ (where 1 thread = 1 core). So, in this example, if the memory request is 4GB, then the total memory resource the job requires is 32GB. In this environment, tmem and h_vmem should be adjusted to avoid significant queueing times. 
• If the application has an option to execute using more than one thread or core, the number given should match that in the environment request. The variable NSLOTS can be used to achieve this. The application must support multi-threading. 
• Please do not request a parallel environment of one core. Apart from the unnecessary scheduler overhead this creates, it does not make sense!
• #$ -pe mpi 8  “[Request a] parallel environment, message-passing interface [with] 8 [cores]”. Use this environment to run multi-threaded or OpenMPI applications across multiple compute nodes. The same factors that apply to SMP environments as described previously, also apply to MPI environments. 
• MPI environments are particularly suited to workloads that require a substantial compute resources e.g. over 200 cores, so while they can be used run much smaller workloads, there is some additional complexity in running MPI-based applications and this is rarely necessary when the equivalent SMP environment is available. 
• Please contact us if you are unfamiliar with running MPI applications and specifically need to do this.
• The GPU flag takes true as an argument: #$ -l gpu=true. This will schedule a job for a GPU node and request 1 GPU. NOTE: There is no gpu=false equivalent. If you no longer wish to use a GPU, simply remove the gpu=true flag. To request multiple GPUs, you must to include the -pe gpu flag as follows: #$ -pe gpu n where n is the number of GPUs requested.
• Note that the -pe gpu flag augments the way the scheduler interprets tmem and h_vmem. These values will be multiplied by n, since the scheduler interprets them as being ‘per thread’ (or per GPU in this case - as for parallel environments). For example, a job requiring 4GB of memory and 2 GPUs would have a submission script containing: #$ -l tmem=2G, #$ -l gpu=true, #$ -pe gpu 2, #$ -R y and a job requiring 4GB of memory and a single GPU would include: #$ -l tmem=4G, #$ -l gpu=true. 
• NOTE: h_vmem should be omitted for certain GPU jobs, specifically PyTorch and Tensorflow, as it causes memory allocation errors on the GPU.
• The flags for requesting GPUs also apply when requesting interactive GPU sessions. 
• You can combine the -pe gpu n directive with a qrsh request to test interactively across multiple GPUs: qrsh -l tmem=4G,gpu=true,h_rt=0:60:0 -pe gpu 2 
• There is a fast test queue for interactive work with a short runtime (up to one hour) to enable you to test code quickly without having to wait: qrsh -l tmem=14G,gpu=true,h_rt=0:30:0 
• Additionally, please only start up to two qrsh sessions per user (except those with group GPU nodes where you can request as many sessions as there are slots available). 
• You no longer have to run the CUDA_VISIBILITY.sh script when running interactive/test sessions. The scheduler now handles all environment variables and the same applies with respect to manually setting the CUDA_VISIBLE_DEVICES variable. This variable controls which cards are visible and is set automatically. Please do not attempt to modify it from within your code, nor specify a GPU device index to use.
When performing large-scale job submissions, the I/O profile of the jobs must be taken into account, to avoid creating I/O bottle-necks that will slow-down job execution for everyone. 
• Jobs performing significant I/O on input or output files, should be run from the local disk on the compute node. This can be requested in the job script using the #$ -l tscratch=nG option, where n is the amount of space requested in gigabytes. 
• In the job script, include a command similar to: mkdir -p /scratch0//${JOB_ID}[.${SGE_TASK_ID}(array job)] 
• This is to prevent multiple jobs running on the same node from overwriting each other. 
• Remember scratch space is for transient job data only and the data could be deleted without warning.
• When the application has executed, please delete the data from within the job script. To ensure this happens, incorporate a trap statement. For example, after the application command line: function finish { rm -rf /scratch0//${JOB_ID}[.${SGE_TASK_ID(array job)] } trap finish EXIT ERR INT TERM 
• This will run the finish function whenever the job exits, regardless of whether it finished successfully or not.
• You can request and interactive session on the cluster using the qrsh command. 
• The qrsh command requires a memory request using the tmem and h_vmem flags, just as in a job script e.g. “qrsh -l tmem=2G,h_vmem=2G [-pe smp 2 (for multicore jobs adjusting the memory request accordingly)]” 
• As with job memory requests, the more memory you request for a qrsh session, the longer it may take for a session to be granted. If the requested resources are not available, the session will be refused. 
• Unlike job scripts, a time request is not required; a qrsh session will last as long as needed.
• Applications available to run on the cluster can be found in the directory /share/apps 
• Within /share/apps there are directories for specific disciplines such as economics (/share/apps/econ) and biosciences (/share/apps/genomics). Some groups have their own directories, which can only be accessed by group members. 
• Users can request applications to be installed centrally within /share/apps or are free to install applications in their home directories and project stores. 
• Certain installed applications need an environment to be setup before they will run correctly. For those where this is the case, there are source files in /share/apps/source_files Source the appropriate file (e.g. source /share/apps/source_files/metaMix.source) to set-up the application environment.. 

#### Using Python – Installing Packages
• There are additional example job scripts in /share/apps/examples
We have numerous versions of python3 (and the last version of python2) in /share/apps. To access any version of python, you must first “source” the source files for the python version you want to use, from /share/apps/source_files/python. For example, if you want to use python-3.9.5-shared you must first run the command: source /share/apps/source_files/python/python-3.9.5.source This will add /share/apps/python-3.9.5-shared/bin to your PATH and /share/apps/python-3.9.5-shared/lib to LD_LIBRARY_PATH

You can install packages for a particular python version without having to request us to install them centrally. To do this, first source the python version you want to install packages into (e.g. version 3.9.5). Then, determine if a package is installed by running the command pip3 list | grep If the package is not found, it can be installed locally in your home directory, using the command: python3 -m pip install --user This will install into the .local directory in your home directory, under 3.9.5. The package will be included with the centrally-installed site-packages of the python version you used to install it, when you run that version of python.

#### Applying for Project Space
• Information about storage on the cluster can be found at: hpc.cs.ucl.ac.uk/data-storage. The storage request form is available here and this should be completed to apply for project space. 
• This space is for processing research data on the cluster and is effectively mandatory. 
• Home directories have strictly limited space and should not be used for data processing. 
• You must confirm with your PI that you have permission to request the amount of have project space you are requesting.

#### File Management
Points to consider when performing large-scale job submissions: 
• Use the #$ -j y flag, so GE only writes a .o file for each job/task. 
• Running pipelines can typically result in a large number of output files being written to a common directory. This can slow the response of the file-system, which particularly impacts on data backups when they are run. Many of these files may be intermediate steps in the pipeline and can be deleted when the pipeline has completed, but otherwise, it is recommended that they be written to a tar archive. This reduces the file count, so relieving the impact on the file system. 
• Note that the quota applied to project space on the CS cluster, includes a file count limit. Consequently, large numbers of small files can cause exceeded quota errors, even though relatively little disk space has been used. 
• It is recommended that, where possible, large files such as those containing very large numbers of DNA sequences, should be processed in compressed form, if the application supports this.

#### Help Requests
Please consider the following points when requesting help with job issues: 
Provide as much information as possible about the job that is experiencing the issue. This should include the path to the job script and job number, the application being run and any messages from GE or the application, before the issue was encountered. 
• Include the “hostname” and “date” commands in the job script, so the node accepting the job and the start time can be determined. Sometimes, an issue with the node or the cluster can cause a job to error. 
• There is more information about the CS cluster on the web-site: hpc.cs.ucl.ac.uk [username/password: “hpc”/“comic”] .

#### Example Job Scripts
**Single Core Job**
```
# Single-core job script example. 
#$ -S /bin/bash #$ -l h_rt=0:10:0 
#$ -l h_vmem=2G,tmem=2G 
#$ -wd /home/users/dg/work 
#$ -j y 
#$ -N md_single 
# Commands to be executed follow. hostname date /home/users/dg/SGEtest/md_single
```
**Array Job**
```
# Array job script example. 
#$ -S /bin/bash #$ -l h_rt=0:10:0 
#$ -l h_vmem=2G,tmem=2G 
#$ -wd /home/users/dg/work 
#$ -j y #$ -N md_single 
#$ -t 1-100 # Array task range. # SGE_TASK_ID is set to a number in the array task range. 
# Commands to be executed follow. hostname date /home/users/dg/SGEtest/md_single -log task.${SGE_TASK_ID}
```
**Multi-Core SMP Job**
```
# Multi-core SMP job script example. 
#$ -S /bin/bash #$ -l h_rt=0:10:0 
#$ -l h_vmem=2G,tmem=2G # Initial memory values. 
#$ -wd /home/users/dg/work 
#$ -j y 
#$ -N md_openmp 
#$ -pe smp 16 # Core count multiplies initial memory values. 
#$ -R y # Reserve (16) cores. 
# Commands to be executed; NSLOTS sets threads to smp core count. hostname date /home/users/dg/SGEtest/md_openmp -threads $NSLOTS
```
**Multi-Core MPI Job**
```
# Multi-core MPI job script example. 
#!/bin/bash 
#$ -S /bin/bash 
#$ -l h_rt=02:00:00 
#$ -l tmem=3.1G,h_vmem=3.1G # Initial memory values. 
#$ -pe mpi 2 # Core count multiplies initial memory values. 
#$ -R y # Reserve (2) cores. #$ -wd /home/dgregory/metaMix # Path to working directory. 
#$ -V # Include user environment. 
# Commands to be executed. hostname date export LD_LIBRARY_PATH=/share/apps/genomics/metaMix/openmpi-1.6.2/lib:$LD_LIBRARY_PATH export PATH=/share/apps/genomics/metaMix/openmpi-1.6.2/bin:$PATH # NSLOTS sets number of processors to mpi core count. # mpirun starts the job a provides the communication interface. mpirun --mca btl_tcp_if_include 128.41.97.0/21 -np $NSLOTS /share/apps/genomics/ metaMix/R-3.5.2/bin/R --slave CMD BATCH --no-save --no-restore Rsubmit.R step.out
```
**GPU Job**
```
# GPU job script example. 
#$ -S /bin/bash 
#$ -l h_rt=0:10:0 
#$ -l tmem=2G # Initial memory value. Note no h_vmem. 
#$ -wd /home/users/dg/work 
#$ -j y 
#$ -N md_gpu 
#$ -l gpu=true 
#$ -pe gpu 2 # GPU count multiplies initial memory value. 
#$ -R y 
# Commands to be executed follow. hostname date /home/users/dg/SGEtest/md_gpu
```
**Multi-Environment Job**
```
# Multi-environment job script example. 
#$ -S /bin/bash 
#$ -l h_rt=240:00:0 
#$ -l tmem=32G,h_vmem=32G 
#$ -wd /home/users/dg/work 
#$ -j y #$ -N run_BEAST_L2 
#$ -pe smp 16 # SMP environment. 
#$ -R y #$ -t 1-12 # Array tasks. 
#Commands to be executed follow. hostname date /share/apps/genomics/BEAST2-2.6.0/bin/beast -threads $NSLOTS l2_bmt_${SGE_TASK_ID}.xml
```

---
### Setting Up an SSH Tunnel

For more information on data transfer, please [see here](https://hpc.cs.ucl.ac.uk/data-transfer/).

In order to ssh to the cluster from outside the CS network in one command you will have to setup a ‘jump’ through a gateway node.

To ssh through a gateway server to a cluster login node use:

`ssh -l <username> -J <username>@<gateway> <login node>`

e.g.

`ssh -l alice -J alice@tails.cs.ucl.ac.uk wise.cs.ucl.ac.uk`

If you don’t want to do this every time you connect to the cluster you can edit the file `~/.ssh/config` and add the following:

`Host <login node>   User <your_CS_username>   HostName <login node>   ProxyJump <username>@<gateway>:22`

e.g.

`Host wise   User alice   HostName wise.cs.ucl.ac.uk   ProxyJump alice@tails.cs.ucl.ac.uk:22`

You can then ssh to wise and this configuration will always be used.

### Port Forwarding with SCP

If you are trying to copy data to and from the cluster from outside the Computer Science department, it might be necessary to set up a port forward so that you connect to a login node via a jump node.  The command to set this up is

`ssh -L 2222:<login node>:22 <username>@<gateway>`

For example, if your username was alice and you wanted to connect to the login node called ‘comic’ via the tails gateway server, the command would be:

`ssh -L 2222:comic.cs.ucl.ac.uk:22 alice@tails.cs.ucl.ac.uk`

Once you have set this up, anything you send to local port 2222 will be forwarded to port 22 on comic via tails (port 22 being the standard port for ssh).

At this point, leave the previous command (`ssh -L`) running, and open another terminal. You can now scp your data by doing the following:

`scp -P 2222 /path/to/file <username>@localhost:~/path/to/destination`

e.g.

`scp -P 2222 /home/alice/data.txt alice@localhost:/home/alice/data_directory`

This will copy the file data.txt in the user alice’s home directory on their local machine to their home directory on the cluster, via the machine tails.

### Tunelling On a Windows Machine (Using Putty)

Start Putty and select SSH in the tree on the left.  In the remote command enter:

`ssh -L 2222:localhost:22 comic.cs.ucl.ac.uk`

Expand ssh and select tunnels. Then, enter source port `2222` and in destination enter `localhost:22`. Select session at the top of the tree on the left and set up a connection to tails as normal (you may want to save this before connecting to save you have to do this again).  You should now have a tunnel through to comic via tails.  Anything you send to port 2222 on localhost will be sent to comic via the ssh tunnel.

---
### Data Transfer

#### External Downloads

Downloading of data must not be done on compute nodes via interactive sessions or submitted jobs. Downloads of any sort should only be performed on login nodes. Small downloads are fine to do on the standard set of cluster login nodes. Larger datasets should be downloaded on little/large.

#### Data Transfer Nodes

little and large (.cs.ucl.ac.uk) are dedicated data transfer cluster login nodes, with higher outbound network bandwith to facilitate downloads of large datasets. Cluster users do not have access to these nodes by default, please contact us with some information about the transfer you want to do and we shall advise and give access if appropriate. In any event where you have a large amount of data to download (1TB or above), please let us know and we can advise if necessary. Long distance data transfers (transatlantic or similar distances) require some attention to get them working as quickly as possible, please contact us if you need to do one.

#### Data Transfer Softwares

This page contains instructions on using Data Transfer softwares to move data to the cluster. If your preferred method of data transfer is not listed below, feel free to [get in contact with us](https://hpc.cs.ucl.ac.uk/contact-us/). For information on SSH and SCP, please [see here](https://hpc.cs.ucl.ac.uk/ssh-scp/).

#### [Aspera Connect](https://hpc.cs.ucl.ac.uk/data-transfer/#details-0-0)

#### [Aspera CLI](https://hpc.cs.ucl.ac.uk/data-transfer/#details-0-1)

#### [WinSCP](https://hpc.cs.ucl.ac.uk/data-transfer/#details-0-2)

#### [Filezilla](https://hpc.cs.ucl.ac.uk/data-transfer/#details-0-3)

#### [rsync](https://hpc.cs.ucl.ac.uk/data-transfer/#details-0-4)

---
### The Cluster

![](https://hpc.cs.ucl.ac.uk/wp-content/uploads/sites/21/2018/12/Cluster_Basics1.png)

The cluster is made up of two main classes of nodes, **login** nodes and **compute** nodes.You connect to the cluster via login nodes. Once connected to a login nodes, you submit jobs to the scheduler. The scheduler then sends your job to an available node that matches the requirements set. The compute node will then run the job you’ve submitted.

#### Login Nodes

Login nodes (comic, vic, and wise for example) are where users will interact with the scheduler. You connect to them via ssh. For example:

`ssh username@wise.cs.ucl.ac.uk`

Login nodes are to be used for **submitting jobs** and **moving data only**.

**Don’t perform any computation on login nodes!**

#### Access

If you are within the CS network (connected to WIRELESS_PETE, CS_GUEST etc.) you can ssh directly to the CS Cluster login nodes. If you try to login directly and get an error, it is likely that you are not on the CS network.

You can access the CS Cluster from outside the CS network, by first logging into one of our gateway nodes.

These are: **knuckles.cs.ucl.ac.uk** for taught students and **tails.cs.ucl.ac.uk** for staff. To login, use the following command from your terminal:

`ssh username@knuckles.cs.ucl.ac.uk`

using your **CS username**.

The usual format of your username will be your first inital, followed by most of your surname. For example: **Joe Bloggs** would have the username **jbloggs**.

#### Bash

The way you will interact with the cluster is via the command line. The shell you will be using is called **bash**. It is very important that you are comfortable with using bash before you can utilise the cluster properly.

Click [here](https://swcarpentry.github.io/shell-novice/) for a link to a basic bash tutorial. Once you are comfortable with these concepts, the intermediate tutorial is [here](https://carpentries-incubator.github.io/shell-extras/). The topics covered in these two tutorials are directly relevant to using the cluster effectively.

#### CS Accounts

CS can provide you with a number of accounts, by default you should have a CS **Unix** account and a CS **Windows** account. These start off with the same credentials, however if you change the password of one (which is recommended) the change **won’t** sync automatically.

In addition, these are **unrelated** to your central UCL account, which will not be valid login details on CS systems.

#### Cluster Account

Your CS cluster account credentials start off as a copy of your CS Unix account details. This means that your login details for the cluster will initially be the same as your CS Unix account (at the time of creation).

Similarly, your cluster account is **not** synced with any others, so any password changes you make to your CS Unix will not be reflected in the cluster.

#### Home Directories

Your CS account has a home directory associated with it. This directory is shared between most managed CS systems, and any update will be shared with all other managed systems automatically.

The cluster has the same concept, however your cluster home directory is **not synced** with your main CS home directory. However, it is shared between **all** cluster nodes (login and compute). You have 50G of space, and it is **not** backed up.

#### Project Stores

Your home directory is limited to 50G, and is not backed up. This is generally not appropriate for computation involving large datasets, so we recommend users to make a request for a project store as soon as it is necessary. If you need a very large amount of storage, please come to room 4.20 Malet Place Engineering Building and we can discuss your project in more detail and find the most effective storage for your needs.

These project stores are usually mounted on the cluster, and can also be directly accessible from managed CS machines. We also allow for access from certain ranges of IP addresses if necessary.

### Submitting Jobs

To use the cluster, you submit ‘jobs’ to the scheduler. This is done using the command **qsub**.

#### qsub

The command qsub can takes a path to a submission script as its argument.

`qsub /path/to/submission/script/`

##### **Submission Scripts**

`#$ -S /bin/bash`

`#$ -l tmem=4G`

`#$ -l h_vmem=4G`

`#$ -l h_rt=10:00:00`

…

`run computation`

The line `#$ -S /bin/bash` tells the scheduler to interpret the rest of the script as a bash script. This again is not strictly mandatory, however it does prevent potential bugs without affecting anything negatively.

The `#$` combination of characters marks the following line as a **scheduler directive**. These are parameters that are passed to the scheduler to define the properties and resource requirements of your jobs.

Below is a more comprehensive list of scheduler directives:

#### [Scheduler Directives](https://hpc.cs.ucl.ac.uk/full-guide/#details-0-0)

  
  

|Flag|Options|Example|Definition|
|---|---|---|---|
|-l|tmem|`#$ -l tmem=4G`|Amount of memory you wish to request.|
||h_vmem|`#$ -l h_vmem=4G`|Hard limit on shell memory usage (ulimit).|
||h_rt|`#$ -l h_rt=1:00:00`|Wall time. Maximum of 2000 hours CPU and 1000 hours GPU|
||gpu|`#$ -l gpu=true`|GPU request.|
||tscratch|`#$ -l tscratch=50G`|Requests scratch space.|
|-S|/bin/bash|`#$ -S /bin/bash`|Shell to be used.|
|-j|y|`#$ -j y`|Merges STDOUT and STDERR.|
|-N|<name>|`#$ -N my_new_job`|Gives your job a name.|
|-t|1-n|`#$ -t 1-10`|Sets a task ‘incrementor’ variable.|
|-tc|n|`#$ -tc 2`|Concurrent task limiter.|
|-wd|/path/|`#$ -wd /home/jbloggs`|Start job from specified working directory.|
|-cwd||`#$ -cwd`|Start job from current working directory.|
|-R|y|`#$ -R y`|Reserves requested resources. Mandatory for parallel environment jobs.|
|-pe|gpu n|`#$ -pe gpu 2`|GPU environment with 2 cards.|
|-pe|smp n|`#$ -pe smp 4`|Parallel environment with 4 cores.|
|-pe|mpi n|`#$ -pe mpi 4`|MPI environment with 4 cores.|
|-pe|mpirr n|`#$ -pe mpirr 4`|MPI environment with 4 cores, allocating cores based on round robin algorithm.|
#### qstat

Once you have submitted a job to the cluster, you can check the state of said job using the **qstat** command.

job-ID  prior   name       user         state submit/start at     queue                          slots ja-task-ID 
-----------------------------------------------------------------------------------------------------------------
6506636 0.00000 testing    jbloggs      qw    21/12/2012 11:11:11                                    1        

The submitted job is now waiting to be scheduled to a node. It has been assigned a job-ID, which you can use with qstat to get more detailed information on a specific job.

`qstat -j <job-ID>`

You’ll get output similar to the following:

[jbloggs@wise ~]$ qstat -j 6506712
==============================================================
job_number:                 6506712
exec_file:                  job_scripts/6506712
submission_time:            Fri Dec 21 11:11:11 2012
owner:                      jbloggs
uid:                        0
group:                      external
gid:                        0
sge_o_home:                 /home/jbloggs
sge_o_log_name:             jbloggs
sge_o_path:                 /usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin ...                      
sge_o_shell:                /bin/bash
sge_o_workdir:              /home/jbloggs
sge_o_host:                 wise
account:                    sge
hard resource_list:         h_vmem=2G,tmem=2G,h_rt=3600
mail_list:                  jbloggs@wise.local
notify:                     FALSE
job_name:                   testing
jobshare:                   0
shell_list:                 NONE:/bin/bash
env_list:                   
script_file:                testing.sub
project:                    csdept
scheduling info:            (Collecting of scheduler job information is turned off)

When the job starts running, you’ll get output similar to the following:

job-ID  prior   name       user         state submit/start at     queue                          slots ja-task-ID 
-----------------------------------------------------------------------------------------------------------------
6506636 0.50612 testing    jbloggs      r     21/12/2012 11:11:11 all.q@burns-609-51.local           1

Another useful piece of information displayed by qstat is the state. The most common states you’ll see are **qw** and **r**. These stand for **queue waiting** and **running** respectively.

Below is a full list of possible states:

#### [States](https://hpc.cs.ucl.ac.uk/full-guide/#details-1-0)

  
  

|State|Meaning|
|---|---|
|qw|Your job is **waiting** to be scheduled.|
|r|Your job is currently **running**.|
|t|Job is in the process of being **transferred** to another compute node.|
|s or S|Job has been **suspended**.|
|h|Job is being **held**.|
|E|Job has an **error**.|
|R|Job has been **restarted (Rr)**, or is waiting to be restarted (Rq).|
|d|Job registered for **deletion**.|
|a or A|Job is in **alarm** state.|
|u|Job is in an **unknown** state.|

  

#### qdel

Use qdel to delete a job that has already been submitted. You can only delete a job that you have submitted.

The command takes a job-ID as an argument:

`qdel <job-ID>`

#### qrsh

The cluster is not a development environment. However, there are a few nodes reserved for testing and profiling.

To use these, you have to make a request to the scheduler. The following command will allocate you an interactive session on one of the test nodes:

`qrsh -l tmem=14G,h_vmem=14G,h_rt=00:59:59`

Once scheduled, you will be on one of the test nodes. You can then run your code via the command line. Running your code does not require you to write a submission script and qsub to the cluster, just run your code from the command line.

There are also two slots reserved on a GPU node for the same purpose, to access these run the following command:

`qrsh -l tmem=14G,h_rt=00:59:59,gpu=true`

#### Cluster Software

There is a central directory of software for use on the cluster. It is located at /share/apps . Please check this directory (and relevant sub-directories) for the software you are trying to use before you ask for it to be installed.

Within /share/apps is a directory called examples. This is where we store source scripts and example submission scripts.

For example, within /share/apps/source_files/python there is a file called python-3.8.5.source, which contains the following:

#Python 3.8.3 Source
export PATH=/share/apps/python-3.8.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.8.5-shared/lib:$LD_LIBRARY_PATH

This sets two bash variables (`PATH` and `LD_LIBRARY_PATH`), which will allow you to simply type `python` to run python v3.8.5 .

Please feel free to ask us about specific software and the corresponding environment setups you will need to run them. Either come to see us in person in room 4.20 Malet Place Engineering Building, or email cluster-support@cs.ucl.ac.uk

---
## Job Submission

We use Sun Grid Engine (SGE) as the job scheduler, the manual can be found [here](https://www.rocksclusters.org/assets/usersguides/roll-documentation/sge/6.2/).

### Submitting a Job

To submit a job run the following command from a login node:

`prompt$ qsub myJobScript.sh`

You will get confirmation that the job has been submitted successfully as well as the a job number e.g.

`Your job  1234567 ("MyTESTJOBNAME") has been submitted`

Once the job has been submitted you can get extra info about the job with:

`prompt$ qstat -j <jobid>`

This will show you the status of the job.

`qstat` by itself will show you the status of your whole queue. A status of `r` indicates that the job is running, `qw` indicates the job is waiting to run and `Eqw` indicates an error.

Some example scripts are stored in: /share/apps/examples

Please also take time to look at the other sections of this website particularly [File Systems & Storage](https://hpc.cs.ucl.ac.uk/data-storage/) and [Cluster Etiquette](https://hpc.cs.ucl.ac.uk/cluster-etiquette/)

### Job Script

A simple example of a job script along with an explanation can be seen below:

`#$ -l tmem=2G   #$ -l h_vmem=2G   #$ -l h_rt=1:0:0   #$ -S /bin/bash   #$ -j y   #$ -N MyTESTJOBNAME   hostname   date   sleep 20   date`

The #$ will be interpreted as command flags for the scheduler.  You must include at least the first 3 command flags in all of your scripts except if you are using [Tensorflow or PyTorch](https://hpc.cs.ucl.ac.uk/tensorflow_and_pytorch/).  Beneath the command flags should be your script.  The command flags  `-l tmem=2G` and `h_vmem=2G` are used to request RAM, you should amend the amount requested as appropriate.  `-l h_rt=x:x:x` is the requested run time of the job in the format hr:min:sec. If you exceed either `h_rt` or `h_vmem` your job will be stopped.

### Array Jobs

The scheduler will store the task number in an environmental variable called `$SGE_TASK_ID` you can use this to reference the data that you want to process. You can use this to make each task within an array job refer to different pieces of data.

For example:

`#$ -l tmem=2G   #$ -l h_vmem=2G   #$ -l h_rt=1:0:0   #$ -S /bin/bash   #$ -N MyTESTJOBNAME   #$ -t 1-1000   hostname   date   ~/programs/process_data.sh "~/data/mydata${SGE_TASK_ID}"   date`

This will submit an array job of 1000 tasks, and each task will run ~/programs/process_data.sh, and process a different file.  i.e Task 1 will process ~/data/mydata1 and task2 will process ~/data/mydata2 etc.

You can also use `SGE_TASK_ID` to refer to lines in a file, say if you have a datapoint on each line.

This can be done in a few different ways, for example:

`sed -n ${SGE_TASK_ID}'{p;q}' input.data`

The `-n` flag suppresses the ‘normal’ output of `sed` with a `p` command, which is to print the file you give it. The `q` quits `sed` as soon as the specified line is reached, which minimises unnecessary processing of the input file.

or

`head -${SGE_TASK_ID} input.data | tail -1`

`head -N` prints the first N lines. `tail -N` prints the last N lines. In this case, you are running `tail -1` on a subset of the initial file (output of the `head` command), which has the Nth line as the final entry.

### SMP

SMP (Symmetric Multiprocessing) is a parallel environment for running multi-threaded jobs within a single node.  Before you request an SMP environment you should ensure that your code is capable of running multi-threaded.  To request an SMP environment use:

`#$ -pe smp 5`

Where 5 is the number of slots (amend this as needed)

You should always enable resource reservation on the job with

`#$ -R y`

This will stop smaller jobs from jumping ahead and causing the job to queue for longer.

### MPI

MPI (Message Passing Interface) is a parallel environment for running a multi-threaded job using the MPI protocol.  This is typically used for communicating between nodes but can be used within a node as well. You request an MPI environment with:

`#$ -pe mpi 100`

or

`#$ -pe mpirr 100`

where 100 is the number of slots (amend this as needed).  The mpi environment will try to keep the slots close together, spread across as few nodes as possible.  The mpirr environment will do the opposite and will try to allocate the slots as widely as possible across many nodes.  Generally the mpi environment works better in most scenarios.

As with SMP jobs you should enable resource reservation on the job with

`#$ -R y`

### GPUs

The gpu directive takes true as an argument.

`#$ -l gpu=true`

In addition, to request multiple GPUs you need to include the `-pe gpu` directive as follows:

`#$ -pe gpu n`

Where n is the number of GPUs requested. Please note that the `-pe gpu n` directive augments the way the scheduler interprets `tmem` and `h_vmem`. These values will be multiplied by `n`, as the scheduler interprets the values of these directives as being ‘per thread’ (or per GPU in this case).

**NB/** `h_vmem` can be omitted for certain GPU jobs (PyTorch and Tensorflow for example).

For example, a job requiring 4G of memory and 2 GPUs would have a submission script containing:

`#$ -l tmem=2G`  
`#$ -l gpu=true`  
`#$ -pe gpu 2`  
`#$ -R y`

and a job requiring 4G of memory and a single GPU:

`#$ -l tmem=4G`  
`#$ -l gpu=true`

The environment variable that controls which cards are visible ( `CUDA_VISIBLE_DEVICES` ) is set for you. Please do not attempt to modify `CUDA_VISIBLE_DEVICES` from within your code, nor specify a GPU device index to use.

There is also a maximum runtime on GPU jobs of **90 days**. If you wish to run jobs for longer than this, please [contact us](https://hpc.cs.ucl.ac.uk/contact-us/).

The NVIDIA drivers are updated regularly, and are currently at version **450.51**.

All GPUs are now set to persistent mode and compute exclusive mode.

### Interactive Sessions (qrsh)

You no longer have to run the `CUDA_VISIBILITY.sh` script when running interactive/test sessions. The scheduler now handles all environment variables for you, and the same applies with respect to manually setting the `CUDA_VISIBLE_DEVICES` variable.

The changes to the flags for requesting GPUs apply to requesting interactive sessions.

You can combine the `-pe gpu n` directive with a qrsh request to test interactively across multiple GPUs:

`qrsh -l tmem=4G,gpu=true,h_rt=3600 -pe gpu 2`

There is a fast test queue for interactive work with a short runtime (up to one hour) to enable you to test code quickly without having to wait:

`qrsh -l tmem=14G,gpu=true,h_rt=0:30:0`

Additionally, please only start up to **two qrsh sessions per user** (except those of you with group GPU nodes where you can request as many sessions as there are slots available).

### Specifiying GPUs

**Specifiying a GPU (especially higher end ones) can vastly increase your queue time,** **and is unnecessary in most cases**.

To request a specific GPU, you can add the following to your submission scripts:

`#$ -l gpu=true,gpu_type=<type>`

Where type is one of the following:

`gtx1080ti` for the GTX 1080ti cards (**11GB**).

`titanx` for the Titan X cards (**12GB**).

`rtx2080ti` for the RTX 2080ti cards (**11GB**).

`rtx4070ti` for the RTX 4070ti Super cards (**16GB**).

`rtx4090` for the RTX 4090 cards (**24GB**).

`p100` for the P100 cards [restricted access only] (**16GB**).

`v100` for the V100 cards [restricted access only] (**16GB**).

`rtx6000` for the RTX 6000 cards [restricted access only] (**24GB**).

`rtx8000` for the RTX 8000 cards [restricted access only] (**48GB**).

`a100` for the A100 cards [restricted access only] (**40GB**).

`a100_80` for the A100 80GB variant cards [restricted access only] (**80GB**).

`a100_dgx` for the A100 cards housed in DGXs [restricted access only] (**40GB**).

So, for example:

`#$ -l gpu=true,gpu_type=gtx1080ti`

will attempt to schedule you only on nodes with a 1080ti on them.

You can also use logical operators to combine multiple types, for example if you wanted only cards with 12GB of memory:

`#$ -l gpu=true,gpu_type=(titanxp|titanx)`

Or if you wanted to omit all cards with less than 12GB of memory:

`#$ -l gpu=true,gpu_type=!(gtx1080ti|rtx2080ti)`

#### NVIDIA CUDA/cuDNN

Versions of CUDA are installed in the default location on the GPU nodes of /usr/local/cuda-<version>  with a symlink from /usr/local/cuda to the most current version. NB/ cuDNN is NOT installed on the /usr/local/ version of CUDA. If you wish to use cuDNN there are also all available within the CUDA versions installed in /share/apps/. Each CUDA directory has its corresponding cuDNN versions installed within it.

  

|CUDA version|cuDNN version|
|---|---|
|7.5|5.1|
||6.0|
|8.0|5.1|
||6.0|
||7.0.4|
||7.0.5|
|9.0|7.0.4|
||7.0.5|
|9.1|7.0.5|
|9.2|7.2.1|
|10.0|7.4.2|
||7.6.3|
|10.1|7.5.0|

#### Torch DDP

There are times where DistributedDataParallel (DDP) calls will hang when using a combination of NCCL backend and PyTorch, especially on cards with NVLinks across them. Try adding the following:

`export NCCL_P2P_DISABLE=1`

to your submission scripts before you call `python3`. If you are encountering further problems, please contact us.


Storage on the Cluster
 

To make best use of the cluster it is important to be aware of how the file systems attached to the cluster are arranged.  If you read/write simultaneously from 100+ nodes to a server with one disk you will get, at best, the speed of the disk divided by the number of nodes. In reality your performance will be much worse.

In an ideal world your job should be CPU bound and not I/O bound. That is to say you don’t want a processor sitting idle while it waits to get data from a disk.

 

Home Directory
Your home directory is located at /home/<username>. This is separate from your main CS file store and is not backed up. Your home folder has a quota of 50GB and is hosted on a BlueArc Titan 4080 clustered NAS. Please do not use your home directory to store data in the long term.

 

Project Stores
For example: /cluster/project<n>/<projectname>

This is specially requested project space and is the only storage on the cluster that is backed up. Space here is available on request. You should request as many project spaces as you need, one per project.

Certain limited central file systems are available directly on the cluster.

Where they are available the general file stores from /cs/research/…/…/ will be mounted under

/SAN/<filestorename>

Click here to fill in a storage request form.

 

Node Local Scratch Space
Each node has its own local disk or SSD, with typically around 200GB of space available.

This is mounted on /scratch0 and is typically the best place for large amounts of data to be accessed from by your job. N.B. Scratch means just that it will get deleted on a regular basis. Note that /scratch0 is different for each node.

To use /scratch0 add #$ -l tscratch=<n>G to your submission script. Additionally, before you move any data to scratch, create a directory at the top level of /scratch0 with your username and job ID. For example:

mkdir -p /scratch0/alice/$JOB_ID

If you are running an array job, add the task id to the directory name too:

mkdir -p /scratch0/alice/$JOB_ID.$SGE_TASK_ID

This is to prevent multiple jobs running on the same node from overwriting each other.

 

Please remember that scratch space is for transient job data only, and the data could be deleted without warning.

When your job is finished, remember to delete said data within your job script. To ensure this happens you can incorporate a trap statement into your submission script. For example, with a non-array job:

 

function finish {
    rm -rf /scratch0/alice/$JOB_ID
}

trap finish EXIT ERR INT TERM

...
This will run the finish function whenever the job exits (regardless of whether the job finished successfully or not).

 

Reference Datasets
 

We are trying to co-ordinate some of the reference sets available on the cluster to avoid duplication.

There is a /share/ref folder with links to all the sets and an index in the README.TXT

If you are creating a reference set of public data or a closed reference set for UCL or group use only let us know and we can add it here.

The list of current sets will soon be made available.

 

Please do not be afraid to discuss what your storage needs are. Also please remember to request money for storing you data on your grants.