{{- define "llmman-csi.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "llmman-csi.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "llmman-csi.selectorLabels" -}}
app.kubernetes.io/name: {{ include "llmman-csi.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "llmman-csi.labels" -}}
{{ include "llmman-csi.selectorLabels" . }}
app.kubernetes.io/version: {{ .Values.image.tag | default .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end -}}

{{/* The Secret holding the registry credentials, or empty for none. */}}
{{- define "llmman-csi.registrySecret" -}}
{{- if .Values.registryAuth.existingSecret -}}
{{- .Values.registryAuth.existingSecret -}}
{{- else if .Values.registryAuth.dockerConfigJson -}}
{{- printf "%s-registry-auth" (include "llmman-csi.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/* Where the plugin's socket lives on the host, and where the kubelet finds it. */}}
{{- define "llmman-csi.pluginDir" -}}
{{- printf "%s/plugins/%s" (trimSuffix "/" .Values.kubeletDir) .Values.driverName -}}
{{- end -}}
