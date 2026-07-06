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


