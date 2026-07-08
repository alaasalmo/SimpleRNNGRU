# Run Job on Kubernetes

<img src="img\project-with Kubernetes.png">


## Start minikuke

```
minikube start --driver=docker
```

Note: In case of any error run: "minikube delete --all --purge" and start the minikube again

## Get all pods to check Kubernretes

```
 kubectl get pods -A
```

We need to open 2 terminals

```
# Terminal 2 — keep this running the whole time
minikube mount "C:\alaa\tmu\project\finanace-example\docker\docker\data:/data"

# Back in Terminal 1 — verify the bridge worked
minikube ssh -- ls /data/input

```

Build configmap for all workers

<a href=configmap-multi-workers.yaml>configmap-multi-workers.yaml</a> 

```
kubectl apply configmap-multi-workers.yaml

```

Build the services for all workers

<a href="services-multi-workers.yaml">services-multi-workers.yaml</a>

```
kubectl apply services-multi-workers.yaml

```

Build PV and PVC

<a href="pv-pvc-multi-workers.yaml">pv-pvc-multi-workers.yaml</a>

```
kubectl apply pv-pvc-multi-workers.yaml

```

Add the local path to minikube

```
# 2. Bridge your Windows folder in — leave this terminal open
minikube mount "C:\alaa\tmu\project\finanace-example\docker\docker\data:/data"
```

After we build the image in the docker section, we have to add it to minikube to use it in the jobs

```
#docker build -t sentiment-twitter-worker .
minikube image load birnngru-twitter:latest
```


```
kubectl delete job bitsimplernn-worker-0 bitsimplernn-worker-1 bitsimplernn-worker-2

```


