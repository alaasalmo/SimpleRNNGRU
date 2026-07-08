## Financial News Sentiment Analysis Using a Distributed Training System


<center><img src="img\header-project.png"></center>

The project will consist of three phases

<img src="img\introduction-system.png">

<b>1- Choose the Models for Deeplearning and improve them (SimpleRNN and GRU)</b>

The first phase focuses on developing and improving the sentiment analysis models. During this phase, we compare the performance of the two models before and after applying several improvement techniques. These techniques include bidirectional architectures, text augmentation, and class weighting. The models are evaluated and compared using accuracy, precision, recall, F1-score, and support.

<img src="img\analysis.jpg">

<a href="analysis"> Data Analysis and Result (Simple RNN & GRU model)</a>

<b>2- Choose the improved models and build them in container</b>

After selecting the optimal models, tuning the hyperparameters, applying dropout, and building the final architectures, the next step is to containerize the models using Docker. To do this, Docker Desktop must be installed on the host machine.

Required software:


| Seq                |      Explanation       |  
|--------------      |:----------------------:|
| 1- Analysis  steps | <a href="https://github.com/alaasalmo/SimpleRNNGRU/tree/main/analysis">Data Analysis and Result</a>        |
| 2- Result & Comparsion | <a href="https://github.com/alaasalmo/SimpleRNNGRU/blob/main/analysis/pipeline/README.md">Models and Result</a> |
| 3- Model comparsion | <a href="https://github.com/alaasalmo/SimpleRNNGRU/tree/main/analysis/model-comparision"> SimpleRNN and GRU comparsion</a> |
| 


Docker Desktop: https://www.docker.com/products/docker-desktop

Docker is supported on both Windows and Linux operating systems. In this implementation, Docker Desktop will be installed on a Windows host to build and manage the containers.

<img src="img\docker-drawio.png">

<a href="docker">Simple RNN & GRU model container base</a>

<b>3- Choose the containers with Kubernetes and scale up the models</b>

<b>Required sofware:</b>

<b>For server as one node:</b> <a href="https://minikube.sigs.k8s.io/docs/start/?arch=%2Fwindows%2Fx86-64%2Fstable%2F.exe+download">Minikube</a> 

<b>For client tool:</b> <a href="https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/">Kubectl</a> 

We will use Minikube instead of Kubernetes. With Minikube, we can run it on Windows and we can use the host machine as one node.


<a href="kubernetes">kubernetes</a>

<b>Result and conclusion</b>


Error, Overfitting

Use Macro-F1 and Accuracy (hiesght)



