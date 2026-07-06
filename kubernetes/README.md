# Run Job on Kubernetes

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

Build the services for all workers

<a href="services-multi-workers.yaml">services-multi-workers.yaml</a>

Build PV and PVC

<a href="pv-pvc-multi-workers.yaml">pv-pvc-multi-workers.yaml</a>