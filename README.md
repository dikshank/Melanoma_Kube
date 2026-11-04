"# melanoma" 
create the namespace first: kubectl create namespace melanoma-production
apply the frontend: kubectl apply -f frontend/deployment.yaml

# Apply failed so Checks: kubectl get pods -n melanoma-production
            C:\Users\A110718\PROJECTS\melanoma\k8s>kubectl get pods -n melanoma-production
            NAME                                   READY   STATUS         RESTARTS   AGE
            frontend-deployment-5b88d9957d-lgpgs   0/1     ErrImagePull   0          65s
            frontend-deployment-5b88d9957d-pkqjv   0/1     ErrImagePull   0          65s
            frontend-deployment-5b88d9957d-xfdwg   0/1     ErrImagePull   0          65s


            because we have not created the image for this, (note that etcd is not the place where we have the images it is only for text data)
            Images will be in /var/lib/docker or /var/lib/containerd)


            check which runtime node uses: kubectl get nodes -o wide and the look at the container-runtime column

            build the image directly on the minikube node: minikube image build -t frontend-image:v1 ../frontend


            It did not work so check minikube status
            C:\Users\A110718\PROJECTS\melanoma\k8s>minikube status
                minikube
                type: Control Plane
                host: Stopped
                kubelet: Stopped
                apiserver: Stopped
                kubeconfig: Stopped
            so start minikube first: minikube start --cpus=5 --memory=12g
            now check the status again: minikube status
            check nodes: kubectl get nodes
            check images: kubectl image ls
            check more with: minikube profile list OR type %USERPROFILE%\.minikube\profiles\minikube\config.json | findstr /I "cpus memory driver" to check the resources

            try again: minikube image build -t frontend-image:v1 ../frontend
            check with: minikube image ls or minikube image ls | findstr frontend

            create again: kubectl create namespace melanoma-production
            apply again: kubectl apply -f frontend/deployment.yaml


            C:\Users\A110718\PROJECTS\melanoma\k8s>kubectl get pods -n melanoma-production
                NAME                                   READY   STATUS   RESTARTS      AGE
                frontend-deployment-66f9f9d7c8-c9drt   0/1     Error    1 (11s ago)   17s
                frontend-deployment-66f9f9d7c8-jbqq6   0/1     Error    1 (11s ago)   17s
                frontend-deployment-66f9f9d7c8-xqg46   0/1     Error    1 (11s ago)   17s

            good news that it was atleast pulled
            checks: kubectl logs frontend-deployment-66f9f9d7c8-c9drt -n melanoma-production
                    2026/10/11 06:53:50 [emerg] 1#1: host not found in upstream "backend" in /etc/nginx/conf.d/default.conf:22
                    nginx: [emerg] host not found in upstream "backend" in /etc/nginx/conf.d/default.conf:22

                    that means the nginx.conf can not find backend:8000, that means we have to first crate the backed


# Backend
create image: minikube image build -t backend-image:v1 ../backend
check with: minikube image ls or minikube image ls | findstr backend

apply: kubectl apply -f backend/deployment.yaml
check: kubectl get pods -n melanoma-production
and since they are stil pending check: kubectl describe pod backend-deployment-7c9bd664d6-96d8v -n melanoma-production

They will be pending because there is no storage pvc yamls
let's create and apply: kubectl apply -f storage/logs-pvc.yaml and kubectl apply -f storage/mlruns-pvc.yaml and kubectl apply -f storage/data-pvc.yaml


now the pods will still node start as there is no model to load from the mlruns because airflow has not updated anything in that because airflow is not created, so let's create airflow


# Airflow:
create image minikube image build -t airflow-image:v1 ../airflow
check: minikube image ls | findstr airflow

apply airflow-config, airflow-secret, postgres otherwise airflow init will not work, and yes it will handle it if statsd exporter is not applied so just to see what happens I will delibrately apply it later.
kubectl apply -f airflow/airflow-config.yaml         

        C:\Users\A110718\PROJECTS\melanoma\k8s>kubectl apply -f airflow/airflow-config.yaml
        error: resource mapping not found for name: "airflow-config" namespace: "melanoma-production" from "airflow/airflow-config.yaml": no matches for kind "configMap" in version "v1"
        ensure CRDs are installed first
        Change configMap to ConfigMap (capital C and capital M):

kubectl apply -f airflow/airflow-secret.yaml   
    checks: kubectl get secrets -n melanoma-production OR   kubectl describe secret airflow-secret -n melanoma-production    or kubectl get secret airflow-secret -n melanoma-production -o yaml    

kubectl apply -f airfloworage/postgres.yaml   
        C:\Users\A110718\PROJECTS\melanoma\k8s>kubectl apply -f airflow/postgres.yaml      
        secret/postgres-secret created
        service/postgres created
        resource mapping not found for name: "postgres-data" namespace: "melanoma-production" from "airflow/postgres.yaml": no matches for kind "PersistentVolumeClaim" in version "apps/v1"
        ensure CRDs are installed first
        Error from server (BadRequest): error when creating "airflow/postgres.yaml": Deployment in version "v1" cannot be handled as a Deployment: strict decoding error: unknown field "spec.template.spec.containers[0].livenessProbe.exec.initialDelaySeconds", unknown field "spec.template.spec.containers[0].livenessProbe.exec.periodSeconds", unknown field "spec.template.spec.containers[0].readinessProbe.exec.periodSeconds"

        fixed 
        check: kubectl get pods -n melanoma-production | findstr postgres
        and kubectl get svc -n melanoma-production
        and kubectl get endpoints postgres -n melanoma-production
        and kubectl exec -it postgres-7d49599479-7v8vq -n melanoma-production -- psql -U airflow -d airflow


Now let's apply the airflow-init-job
kubectl apply -f airflow/airflow-init-job.yaml
    check: minikube image ls | findstr airflow
            kubectl get jobs -n melanoma-production
            kubectl get pods -n melanoma-production -l job-name=airflow-init-job
            kubectl describe job airflow-init-job -n melanoma-production
            kubectl logs -n melanoma-production -l job-name=airflow-init --tail=100 -f
            kubectl wait --for=condition=complete job/airflow-init -n melanoma-production --timeout=300s

            C:\Users\A110718\PROJECTS\melanoma\k8s>kubectl get pods -n melanoma-production -l job-name=airflow-init    
            NAME                 READY   STATUS                       RESTARTS   AGE
            airflow-init-49jg5   0/1     CreateContainerConfigError   0          5m16s

            check: kubectl describe pod airflow-init-49jg5 -n melanoma-production
            the error was a typo

            we first have to delete this because JOB templates are immutatble : kubectl delete job airflow-init -n melanoma-production 
            kubectl apply -f airflow/airflow-init-job.yaml

            check again: kubectl apply -f airflow/airflow-init-job.yaml
            AND kubectl describe job airflow-init -n melanoma-production

let's apply webserver
kubectl apply -f airflow/airflow-webserver.yaml
    check:
        kubectl get pods -n melanoma-production | findstr webserver
        lanoma-production
        kubectl get pods -n melanoma-production | findstr webserver