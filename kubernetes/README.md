# Run Job on Kubernetes

<img src="img\project-with Kubernetes.png">


## Start minikuke

Start Kubernetes wtih 4 cpus and 14 GB

```
minikube start --driver=docker --memory=14000 --cpus=4

```

Note: In case of any error run: "minikube delete --all --purge" and start the minikube again

## Get all pods to check Kubernretes

```
 kubectl get pods -A
```

We need to include the images (containers that we built them in the previous section Docker)

Load the images:

1- One image for worker (Chief + workers). In our example we use one container for Chief-worker and 2 containers for workers

```
minikube image load birnngru-twitter:latest
```

2- One image to build a container to do the testing 

```
minikube image load sentiment-predict:latest
```

3- Build one folder inside the minikube to have to the input and output files

In the example, we use the twitter containers, this mean we will not have files in the input. 

```
# Back in Terminal 1 — verify the bridge worked
minikube ssh
sudo su -
mkdir -p /data/input


```
Note: In case of using the Kaggle conteainer we will need to copy the file from local host to the /data/input/

```
kubectl cp "C:\alaa\all-data.csv" <pod-name>:/data/input/

```

4- Build configmap for all workers

After complete the prepation of the folders (input and output) in Kubernetes, we have to build the workers

we have to build:

A- Three workers (one is chief-worker and another are workers).

The main role for chief worker is get data and manage the other workers. Also to distribute the data among the three workers.

The other role for chief-worker is to train the data and collect the data from another workers and put them in one output/checkpoint folder.

The role for another workers are to train data.

The main components in our project-with

- Three workers

- One configmap to keep the confgiuration

- Three services for each worker, we will need one services

- Three jobs. For each worker, we will need one job

- Data storage for the containers that point to /data/input and output.

<b>PersistentVolume (PV) </b> is a physical storage resource in the cluster, and <b>PersistentVolumeClaim (PVC)</b> is a user's request to use that storage


<b>Configmap:</b>

<a href=configmap-multi-workers.yaml>configmap-multi-workers.yaml</a> 

```
kubectl apply configmap-multi-workers.yaml

```

<b>Services for all workers</b>

<a href="services-multi-workers.yaml">services-multi-workers.yaml</a>

```
kubectl apply services-multi-workers.yaml

```

Build PV and PVC

<a href="pv-pvc-multi-workers.yaml">pv-pvc-multi-workers.yaml</a>

```
kubectl apply pv-pvc-multi-workers.yaml

```

Build the job to run the worker.

```
kubectl apply -f apply-jobs-multi-workers.yaml 
```

<a href="apply-jobs-multi-workers.yaml">apply-jobs-multi-workers.yaml</a>


Check the run:

```
kubectl get jobs
```

<img src="img\output-worker.png" >

<img src="img\output-worker-2.png" >




