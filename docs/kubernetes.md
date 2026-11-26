# Kubernetes

`llmman-csi` is a [CSI](https://kubernetes-csi.github.io/docs/) node driver
that mounts a [ModelPack](https://github.com/modelpack/model-spec) model
artifact into a pod, read-only, as the complete model directory: weights,
`config.json`, tokenizer files, GGUF files, docs, every file at the path the
artifact was packaged with. vLLM, SGLang, llama.cpp and anything else that
reads a model from a directory can point straight at it.

It drops in for [modelpack/model-csi-driver](https://github.com/modelpack/model-csi-driver):
the driver name, `model.csi.modelpack.org`, and the volume attribute,
`model.csi.modelpack.org/reference`, are the same, so a pod written for that
driver runs unchanged. [Compared with model-csi-driver](#compared-with-model-csi-driver)
says what is different and what is not supported.

```yaml
volumes:
  - name: model-volume
    csi:
      driver: model.csi.modelpack.org
      volumeAttributes:
        model.csi.modelpack.org/reference: "registry.example.com/models/qwen3-0.6b:latest"
```

## How it works

Each node runs one DaemonSet pod holding two containers: the Kubernetes
`node-driver-registrar` sidecar, and `llmman-csi`, which serves the CSI
Identity and Node services on a unix socket (there is no controller service
and no staging step). The `llmman-csi` image also holds the `llmman` binary.

When the kubelet publishes a volume, the driver runs
`llmman resolve --dir <reference>`, which

1. pulls the artifact into llmman's OCI store on that node if the node does
   not have it, enforcing the `[verify]` signature policy exactly as
   `llmman pull` does;
2. lays every layer out under `<cache>/<manifest digest>/`, each file at its
   `org.cncf.model.filepath`, unpacking tar layers once, and linking files into
   place so each exists on disk once;
3. prints that directory.

The driver then mounts the directory read-only at the pod's volume path.

Two consequences follow. A model is pulled and unpacked once per node, however
many pods name it; a hundred replicas that start together share one pull. And a
volume costs no disk beyond the model: nothing is copied per pod.

## Requirements

* Kubernetes 1.25 or later (inline CSI volumes are GA from 1.25), on Linux
  nodes with overlayfs, which every container runtime already needs.
* The release namespace must allow privileged pods. The driver bind-mounts
  under the kubelet's pod directories, which needs `CAP_SYS_ADMIN` and mount
  propagation to the host. With Pod Security admission:
  `kubectl label namespace llmman-csi pod-security.kubernetes.io/enforce=privileged`.
* A node directory with room for the models (`storage.hostPath`, default
  `/var/lib/llmman-csi`).

## Install

The chart is in `deploy/helm/llmman-csi`. The image is built from
`csi/Dockerfile` and published with the release as `docker.io/ai/llmman-csi`,
tagged with the release version.

```console
$ helm upgrade --install llmman-csi deploy/helm/llmman-csi \
    --namespace llmman-csi --create-namespace \
    --set image.tag=<release version>
$ kubectl -n llmman-csi get pods
$ kubectl get csinodes -o custom-columns=NODE:.metadata.name,DRIVERS:.spec.drivers[*].name
```

Every node that should mount models appears in `get csinodes` with the driver
listed once its pod is running. The chart creates a `CSIDriver` object
(`attachRequired: false`, `podInfoOnMount: true`, `Ephemeral` and `Persistent`
lifecycle modes, `fsGroupPolicy: None` so the kubelet never changes the
ownership of shared files) and the DaemonSet. It needs no RBAC: the driver
does not call the Kubernetes API.

The values are documented in `deploy/helm/llmman-csi/values.yaml`. The ones you
are most likely to set:

| Value | Default | |
|---|---|---|
| `image.tag` | chart `appVersion` | The release to run. |
| `driverName` | `model.csi.modelpack.org` | Name of the driver and of the `CSIDriver`. |
| `kubeletDir` | `/var/lib/kubelet` | The kubelet's `--root-dir`; change it only if it is not the default. |
| `storage.hostPath` | `/var/lib/llmman-csi` | Where each node keeps pulled models. |
| `registryAuth` | none | Credentials for private registries. |
| `llmman.config`, `llmman.keys` | none | `llmman.conf`, for the signature policy. |
| `env` | none | Extra environment, such as `HF_TOKEN`. |
| `pullTimeout` | `1h` | Longest one pull and unpack may take. |
| `hostNetwork` | `false` | For a registry on the node's loopback. |

## Use

### Inline volumes

Declare the volume in the pod, as in the example at the top, and mount it
wherever the server expects its model. These run the same artifact under
different servers (none of them were run for this document except the
llama.cpp one, see [What has been verified](#what-has-been-verified)):

* [`deploy/examples/vllm-pod.yaml`](../deploy/examples/vllm-pod.yaml)
* [`deploy/examples/sglang-pod.yaml`](../deploy/examples/sglang-pod.yaml)
* [`deploy/examples/llama-cpp-pod.yaml`](../deploy/examples/llama-cpp-pod.yaml)

A pod stays in `ContainerCreating` while its model is pulled. The kubelet gives
up on a call after about two minutes and calls again; the driver keeps the pull
going and the retry joins it, so a large model is not pulled twice.
`kubectl describe pod` shows the failure if the pull fails (an unknown
reference, bad credentials, a signature the policy refuses), with llmman's own
message.

### PersistentVolumes

A pre-provisioned PersistentVolume takes the same attribute, for pods that
claim a volume rather than declare one:
[`deploy/examples/persistent-volume.yaml`](../deploy/examples/persistent-volume.yaml).
There is no dynamic provisioning: you create the PersistentVolume.

### Volume attributes

| Attribute | |
|---|---|
| `<driverName>/reference` | The artifact: any reference `llmman pull` understands. Required. |
| `model.csi.modelpack.org/reference` | The same, accepted under any driver name. |

`<driverName>/exclude-model-weights` and `<driverName>/exclude-file-patterns`,
which model-csi-driver takes, are refused unless they ask for nothing: a volume
that cannot give you what you asked for fails rather than mounting the wrong
thing. The attributes the kubelet adds (`csi.storage.k8s.io/...`) are ignored.

## What is in the volume

The root of the volume is the root of the artifact. A layer's file is at the
path in its `org.cncf.model.filepath` annotation, or, for a tar layer, at the
name each entry has in the archive. That is how modctl unpacks an artifact too,
so what modctl packaged under `tokenizer/` is in `/model/tokenizer/`.

All of it is mounted, docs, code and datasets included, not only what a server
needs. A model directory with `config.json` at its root is what vLLM and
SGLang take; for llama.cpp, `-m` names the GGUF file in it.

Files are readable by everyone, writable by no one, and executable only if the
artifact says so.

### Read-only

The volume is read-only for every pod, whatever its `volumeMount` says. This is
enforced in the filesystem, not only the mount: the volume is an overlayfs with
no writable layer. A read-only bind mount would not hold, because a container
runtime asks for a read-write mount of a volume whose `readOnly` is not set, and
gets one, and the model files are shared with every other pod on the node.
(`llmman-csi` was first written that way, and a pod could rewrite
`config.json` for all the others.) Writes fail with `Read-only file system`.

## ModelPack compatibility

llmman reads every layer the [specification](https://github.com/modelpack/model-spec/blob/main/docs/spec.md)
lists, in every category (`weight`, `weight.config`, `doc`, `code`, `dataset`)
and every encoding:

| Media type suffix | Layer |
|---|---|
| `.v1.raw` | The file itself, placed at its `org.cncf.model.filepath`. |
| `.v1.tar` | A tar archive. |
| `.v1.tar+gzip` | A gzip-compressed tar archive (multi-member streams too). |
| `.v1.tar+zstd` | A zstd-compressed tar archive (every frame, skippable frames included). |

An artifact that modctl or the specification produces and llmman cannot mount
is a bug; please report it.

A tar layer comes from a registry, so it is not trusted. Only regular files
and directories are created. An entry whose path is absolute, contains `..`, a
backslash or a colon fails the layer (modctl refuses the first two as well).
Symbolic and hard links are skipped, as modctl does; device nodes, FIFOs and
sparse files fail the layer. A failed layer fails the mount, and nothing
outside the volume's directory is written.

llmman pulls more than ModelPack artifacts (`hf.co/...`, `s3://...` and the
other sources of `llmman pull`), and those references work as volumes too. Only
ModelPack artifacts carry the compatibility guarantee above.

## Registry authentication

llmman reads registry credentials in Docker's `config.json` format: what
`docker login` and `llmman login` write. Give the driver one through a Secret.
Either let the chart create it:

```yaml
# values.yaml
registryAuth:
  dockerConfigJson: |
    {"auths": {"registry.example.com": {"auth": "<base64 of user:password>"}}}
```

or name a Secret of type `kubernetes.io/dockerconfigjson` that exists in the
release namespace:

```console
$ kubectl -n llmman-csi create secret docker-registry model-registry \
    --docker-server=registry.example.com --docker-username=... --docker-password=...
$ helm upgrade llmman-csi deploy/helm/llmman-csi -n llmman-csi --reuse-values \
    --set registryAuth.existingSecret=model-registry
```

The Secret is mounted into the driver pods and `DOCKER_CONFIG` points at it.
Credential helpers do not exist in the image; use `auths`. For `hf.co/...`
references set `HF_TOKEN` through `env`:

```yaml
env:
  - name: HF_TOKEN
    valueFrom:
      secretKeyRef: {name: hf-token, key: token}
```

Credentials are per node and not per pod: any pod that can declare an inline
volume can have the node pull anything those credentials reach. Restrict who
can create pods with CSI volumes (for example with a ValidatingAdmissionPolicy
on `volumes[*].csi.volumeAttributes`) if that matters in your cluster.

llmman speaks plain HTTP to `localhost` and `127.0.0.1` only. A registry
reachable over plain HTTP anywhere else needs TLS in front of it.

## Signature policy

The driver applies llmman's `[verify]` policy to every volume, including one
whose model is already on the node: being on disk is not being trusted. Put the
policy and the public keys in the chart's values; they are mounted at
`/etc/llmman`.

```yaml
llmman:
  config: |
    [verify]
    default = "off"

    [[verify.trust]]
    pattern = "registry.example.com/models/**"
    keys    = ["keys/models.pub"]
    mode    = "enforce"
  keys:
    models.pub: |
      -----BEGIN PUBLIC KEY-----
      ...
      -----END PUBLIC KEY-----
```

Under `enforce` an unsigned model leaves the pod in `ContainerCreating` with
`... is not signed by a trusted key` in its events. The policy applies to OCI
registry references only; see [verification.md](verification.md).

## Storage

Each node keeps what it has pulled in `storage.hostPath`: llmman's OCI store in
`store/` (the layers as pulled) and the model directories in `cache/`. Models
stay until you remove them: nothing prunes the directory and there are no
quotas, so size the disk for the models you run. A raw layer is on disk once, linked
into the model directory from the store. A tar layer, compressed or not (which
is what modctl writes by default), is on disk twice: as pulled, and unpacked.

Pods keep working if the driver pod restarts: their mounts belong to the
kubelet, not the plugin.

To remove a model from a node, remove it with llmman in the driver pod; that
also frees, at once, what no other model uses. Or delete the
directory when no pod uses anything in it:

```console
$ kubectl -n llmman-csi exec ds/llmman-csi -c llmman-csi -- llmman rm registry.example.com/models/qwen3-0.6b:latest
```

Do not remove a model a running pod mounts: its files disappear from the pod.

## Compared with model-csi-driver

| | llmman-csi | modelpack/model-csi-driver |
|---|---|---|
| Driver name and `<name>/reference` attribute | `model.csi.modelpack.org` and the same attribute | the same |
| Inline ephemeral volumes | yes | yes |
| Pre-provisioned PersistentVolumes | yes | yes |
| Dynamic provisioning (PVC from a StorageClass) | no | yes (`model-image` StorageClass, controller service) |
| Disk per node | one copy per model, shared by every pod | one copy per volume |
| Read-only for every pod | yes, in the filesystem | the pod's `readOnly` |
| P2P acceleration (Dragonfly) | no | yes |
| Per-volume disk quota (`check-disk-quota`) | no | yes |
| `exclude-model-weights`, `exclude-file-patterns` | no, refused | yes |
| HTTP API for dynamic volumes, metrics, tracing | no | yes |
| Plain-HTTP registries (`serverscheme: http`) | `localhost` and `127.0.0.1` only | any, per registry |
| Signature policy for pulls | yes (`[verify]`) | no |
| tar, tar+gzip, tar+zstd layers | all | raw and tar (modctl's decoder matches the `.raw` and `.tar` suffixes) |
| Other sources (`hf.co/`, `s3://`, ...) | yes | no |
| Language | Go plugin running the Rust `llmman` binary | Go |

## What has been verified

Verified on a kind cluster (Kubernetes 1.36) with a local registry: the driver
registering with the kubelet; an inline volume and a pre-provisioned
PersistentVolume with `ReadOnlyMany`; an artifact with layers of every category
and encoding; eight pods naming one reference on a cold node (one pull); the
driver restarting under running pods; a private registry with `registryAuth`;
the `[verify]` policy refusing an unsigned artifact; and llama.cpp's
`llama-server` loading a GGUF from the volume and answering a completion. A
pod whose `volumeMount` is not `readOnly` cannot write.

Not verified: vLLM and SGLang pods (they need GPU nodes; the examples are the
documented way to point them at the volume), SELinux-enforcing nodes (label the
store directory `container_file_t`, as for any hostPath shared with pods), and
nodes other than x86_64.
