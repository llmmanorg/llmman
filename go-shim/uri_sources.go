// uri_sources.go — unified URI-scheme dispatcher and per-source download handlers.
//
// Supported URI schemes (mirrors NVIDIA NIM Model-Free NIM):
//
//   hf://owner/repo[:tag]           HuggingFace Hub      (HF_TOKEN)
//   huggingface://owner/repo[:tag]  alias for hf://
//   ms://owner/repo[:revision]      ModelScope Hub       (MODELSCOPE_API_TOKEN)
//   modelscope://...                alias for ms://
//   ngc://org/team/model[:version]  NVIDIA NGC           (NGC_API_KEY)
//   s3://bucket/prefix              AWS S3 / S3-compat.  (AWS_ACCESS_KEY_ID, ...)
//   gs://bucket/prefix              Google Cloud Storage (GOOGLE_APPLICATION_CREDENTIALS)
//   /absolute/path                  Local directory      (no auth)
//
// All sources store their output as a CNCF ModelPack OCI layout, consistent
// with the rest of llmman's storage format.

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/vbauerster/mpb/v8"

	// AWS SDK v2
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

// dispatchPull routes a URI-scheme reference to the appropriate source handler.
// Returns (handled bool, error). When handled=false the caller falls through to
// the OCI registry routing. progressKey is always the exact, original ref
// the daemon's /api/pull handler was given — not any
// source-specific canonicalized form the handler below might separately
// compute (e.g. pullModelScope's own storeRef) — since that's the exact
// string the Rust side polls llmman_progress with (see progress_state.go).
func dispatchPull(ctx context.Context, ref, layoutDir string) (bool, error) {
	if err := ensureLayout(layoutDir); err != nil {
		return true, fmt.Errorf("init layout: %w", err)
	}
	progressKey := ref

	switch {
	case strings.HasPrefix(ref, "hf://") || strings.HasPrefix(ref, "huggingface://"):
		return true, errHFNotHandledHere(ref)

	case strings.HasPrefix(ref, "ms://") || strings.HasPrefix(ref, "modelscope://"):
		msRef := strings.TrimPrefix(strings.TrimPrefix(ref, "ms://"), "modelscope://")
		return true, pullModelScope(ctx, msRef, layoutDir, progressKey)

	case strings.HasPrefix(ref, "ngc://"):
		return true, pullNGC(ctx, ref, layoutDir, progressKey)

	case strings.HasPrefix(ref, "s3://"):
		return true, pullS3(ctx, ref, layoutDir, progressKey)

	case strings.HasPrefix(ref, "gs://"):
		return true, pullGCS(ctx, ref, layoutDir, progressKey)

	case strings.HasPrefix(ref, "/"):
		return true, pullLocal(ref, layoutDir)
	}

	return false, nil
}

// ---------------------------------------------------------------------------
// Media-type classification (CNCF ModelPack)
// ---------------------------------------------------------------------------

// classifyFile maps a file extension to the appropriate CNCF model layer
// media type. Uses the raw (non-tar) variant because each file is stored as
// its own blob without compression.
func classifyFile(name string) string {
	lower := strings.ToLower(filepath.Base(name))
	ext := filepath.Ext(lower)

	switch ext {
	case ".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ggml",
		".gguf_v2", ".ot", ".engine", ".trt", ".onnx":
		return "application/vnd.cncf.model.weight.v1.raw"
	// ".jinja": a standalone chat_template.jinja file.
	case ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
		".model", ".tiktoken", ".vocab", ".merges", ".spm", ".jinja":
		return "application/vnd.cncf.model.weight.config.v1.raw"
	case ".txt":
		// tokenizer vocab / merges files are config; README is doc
		if strings.Contains(lower, "vocab") || strings.Contains(lower, "merges") {
			return "application/vnd.cncf.model.weight.config.v1.raw"
		}
		return "application/vnd.cncf.model.doc.v1.raw"
	case ".py", ".sh", ".js", ".ts":
		return "application/vnd.cncf.model.code.v1.raw"
	}

	switch {
	case strings.HasPrefix(lower, "readme"), strings.HasPrefix(lower, "license"),
		strings.HasPrefix(lower, "licence"), ext == ".md", ext == ".rst",
		ext == ".pdf":
		return "application/vnd.cncf.model.doc.v1.raw"
	}

	return "application/vnd.cncf.model.doc.v1.raw"
}

// ---------------------------------------------------------------------------
// Generic CNCF ModelPack packager
// ---------------------------------------------------------------------------

// modelPackFile describes one file to be stored in a CNCF ModelPack layer.
type modelPackFile struct {
	localPath    string // absolute path to the file on disk
	relativePath string // path recorded in org.cncf.model.filepath annotation
	mediaType    string // CNCF media type (empty → auto-detected)
}

// downloadToPackFile downloads one file into a temp blob path under
// layoutDir/blobs, reporting progress via a bar added to prog, and returns a
// modelPackFile referencing the downloaded temp path on success — the shared
// inner-loop body of every cloud-source pull below (ModelScope, NGC, S3,
// GCS). open performs the actual GET/GetObject call and returns the
// response body; tmpPrefix namespaces the temp file per source kind (e.g.
// "ms", "ngc") to avoid collisions, and kind labels error messages (e.g.
// "ModelScope", "NGC"). Callers remain responsible for calling prog.Wait()
// once, after the whole download loop finishes (successfully or not).
func downloadToPackFile(prog *mpb.Progress, layoutDir, tmpPrefix, kind, relPath string, size int64, open func() (io.ReadCloser, error), progressKey string) (modelPackFile, error) {
	tmpPath := filepath.Join(layoutDir, "blobs", tmpPrefix+"-"+strings.ReplaceAll(relPath, "/", "_")+".part")
	bar := addLayerBar(prog, "Pulling  "+filepath.Base(relPath), "Pulled   "+filepath.Base(relPath), size, progressKey)

	body, err := open()
	if err != nil {
		bar.Abort(false)
		return modelPackFile{}, fmt.Errorf("%s download %s: %w", kind, relPath, err)
	}

	fh, err := os.Create(tmpPath)
	if err != nil {
		body.Close()
		bar.Abort(false)
		return modelPackFile{}, err
	}
	proxy := proxyOrNop(bar, body, progressKey)
	_, copyErr := io.Copy(fh, proxy)
	proxy.Close()
	fh.Close()
	body.Close()
	if copyErr != nil {
		os.Remove(tmpPath)
		return modelPackFile{}, fmt.Errorf("%s write %s: %w", kind, relPath, copyErr)
	}

	return modelPackFile{localPath: tmpPath, relativePath: relPath}, nil
}

// cloudDownloadClient returns the HTTP client used for downloading file
// content from the cloud sources below (ModelScope, NGC, GCS): no body read
// timeout so large files can transfer without a deadline, but a response
// header timeout still prevents hanging on a stalled server. S3 goes through
// the AWS SDK's own client instead, so it doesn't use this.
func cloudDownloadClient() *http.Client {
	return &http.Client{Transport: &http.Transport{
		ResponseHeaderTimeout: 60 * time.Second,
	}}
}

// downloadItem describes one file to be fetched by downloadItemsAsModelPack.
type downloadItem struct {
	relPath string
	size    int64
	open    func() (io.ReadCloser, error)
}

// downloadItemsAsModelPack runs the shared "download every item with a
// progress bar, then pack the results as a CNCF ModelPack" loop used by every
// cloud pull source (ModelScope, NGC, S3, GCS). tmpPrefix disambiguates
// staging file names between sources (e.g. "ms", "ngc"); kind labels error
// messages (e.g. "ModelScope", "NGC"); noFilesErr is returned verbatim if no
// items were downloaded (each source phrases this error slightly differently).
func downloadItemsAsModelPack(layoutDir, tmpPrefix, kind string, items []downloadItem, storeRef, modelRepo string, noFilesErr error, progressKey string) error {
	prog := newProgressPool(80)
	var packFiles []modelPackFile

	for _, it := range items {
		pf, err := downloadToPackFile(prog, layoutDir, tmpPrefix, kind, it.relPath, it.size, it.open, progressKey)
		if err != nil {
			prog.Wait()
			return err
		}
		packFiles = append(packFiles, pf)
	}
	prog.Wait()

	if len(packFiles) == 0 {
		return noFilesErr
	}

	err := packFilesAsModelPack(layoutDir, storeRef, modelRepo, packFiles, false)
	for _, f := range packFiles {
		os.Remove(f.localPath)
	}
	return err
}

// packFilesAsModelPack writes each file as a raw blob and creates a conformant
// CNCF ModelPack OCI manifest referencing them all.  Reuses the same storage
// primitives as the HF path so the format is identical. verbose additionally
// prints a "stored <path> (<size> bytes)" line per file — used by pullLocal's
// local-directory import, which has no download progress bar to show the
// work happening otherwise.
func packFilesAsModelPack(layoutDir, ref, modelRepo string, files []modelPackFile, verbose bool) error {
	var layers []ocispec.Descriptor

	for _, f := range files {
		mt := f.mediaType
		if mt == "" {
			mt = classifyFile(f.relativePath)
		}

		data, err := os.ReadFile(f.localPath)
		if err != nil {
			return fmt.Errorf("read %s: %w", f.localPath, err)
		}

		desc, err := writeBlob(layoutDir, mt, data)
		if err != nil {
			return fmt.Errorf("store %s: %w", f.relativePath, err)
		}
		desc.Annotations = map[string]string{
			"org.cncf.model.filepath": f.relativePath,
		}
		layers = append(layers, desc)
		if verbose {
			fmt.Fprintf(os.Stderr, "  stored %s (%d bytes)\n", f.relativePath, desc.Size)
		}
	}

	// No HuggingFace-style model card to read a license from for these
	// generic file-based sources (ms://, ngc://, s3://, gs://, a local
	// path) — same "safetensors" format label buildCNCFManifest always
	// used here before modelMeta existed.
	return storeSafetensorsAsOCI(layoutDir, ref, modelRepo, modelMeta{Format: "safetensors"}, layers)
}

// ---------------------------------------------------------------------------
// ModelScope  (ms://)
// ---------------------------------------------------------------------------

// msFile is one entry from the ModelScope tree API.
type msFile struct {
	Name string `json:"Name"`
	Path string `json:"Path"`
	Size int64  `json:"Size"`
	Type string `json:"Type"` // "file" or "tree"
}

func pullModelScope(ctx context.Context, msRef, layoutDir, progressKey string) error {
	// Parse owner/repo[:revision]
	owner, repo, revision := "", "", "master"
	parts := strings.SplitN(msRef, "/", 2)
	if len(parts) != 2 {
		return fmt.Errorf("invalid ModelScope ref %q: expected owner/repo[:revision]", msRef)
	}
	owner = parts[0]
	repoRev := parts[1]
	if idx := strings.LastIndex(repoRev, ":"); idx > 0 {
		repo = repoRev[:idx]
		revision = repoRev[idx+1:]
	} else {
		repo = repoRev
	}

	token := os.Getenv("MODELSCOPE_API_TOKEN")
	endpoint := "https://modelscope.cn"
	if ep := os.Getenv("MODELSCOPE_ENDPOINT"); ep != "" {
		endpoint = strings.TrimRight(ep, "/")
	}

	storeRef := fmt.Sprintf("ms://%s/%s:%s", owner, repo, revision)

	// Fast path: already cached.
	if reportCached(layoutDir, storeRef, owner+"/"+repo) {
		return nil
	}

	// List files via API.
	listURL := fmt.Sprintf("%s/api/v1/models/%s/%s/repo/files?Revision=%s&Recursive=true",
		endpoint, owner, repo, revision)
	req, err := http.NewRequestWithContext(ctx, "GET", listURL, nil)
	if err != nil {
		return err
	}
	if token != "" {
		req.Header.Set("Authorization", "Token "+token)
	}
	apiClient := &http.Client{Timeout: 60 * time.Second}
	resp, err := apiClient.Do(req)
	if err != nil {
		return fmt.Errorf("ModelScope list: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("ModelScope list: HTTP %d", resp.StatusCode)
	}

	var result struct {
		Data struct {
			Files []msFile `json:"Files"`
		} `json:"Data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return fmt.Errorf("ModelScope list decode: %w", err)
	}

	dlClient := cloudDownloadClient()

	var items []downloadItem
	for _, f := range result.Data.Files {
		if f.Type != "file" || !shouldDownloadSafetensors(f.Path) {
			continue
		}
		dlURL := fmt.Sprintf("%s/%s/%s/resolve/%s/%s",
			endpoint, owner, repo, revision, f.Path)
		items = append(items, downloadItem{
			relPath: f.Path,
			size:    f.Size,
			open: func() (io.ReadCloser, error) {
				req2, _ := http.NewRequestWithContext(ctx, "GET", dlURL, nil)
				if token != "" {
					req2.Header.Set("Authorization", "Token "+token)
				}
				r, err := dlClient.Do(req2)
				if err != nil {
					return nil, err
				}
				return r.Body, nil
			},
		})
	}

	return downloadItemsAsModelPack(layoutDir, "ms", "ModelScope", items, storeRef, owner+"/"+repo,
		fmt.Errorf("no model files found in ModelScope repo %s/%s", owner, repo), progressKey)
}

// ---------------------------------------------------------------------------
// NVIDIA NGC  (ngc://)
// ---------------------------------------------------------------------------

// NGC ref format: ngc://org/team/model:version  or  ngc://org/model:version
func pullNGC(ctx context.Context, ngcRef, layoutDir, progressKey string) error {
	// Strip scheme
	path := strings.TrimPrefix(ngcRef, "ngc://")
	apiKey := os.Getenv("NGC_API_KEY")
	if apiKey == "" {
		return fmt.Errorf("NGC_API_KEY environment variable is not set")
	}

	// Parse org/[team/]model:version
	parts := strings.SplitN(path, ":", 2)
	modelPath := parts[0]
	version := "latest"
	if len(parts) == 2 {
		version = parts[1]
	}

	storeRef := ngcRef
	if reportCached(layoutDir, storeRef, modelPath) {
		return nil
	}

	apiBase := "https://api.ngc.nvidia.com/v2"
	pathSegs := strings.Split(modelPath, "/")
	var listURL string
	switch len(pathSegs) {
	case 2: // org/model
		listURL = fmt.Sprintf("%s/models/%s/%s/versions/%s/files",
			apiBase, pathSegs[0], pathSegs[1], version)
	case 3: // org/team/model
		listURL = fmt.Sprintf("%s/models/%s/%s/%s/versions/%s/files",
			apiBase, pathSegs[0], pathSegs[1], pathSegs[2], version)
	default:
		return fmt.Errorf("invalid NGC path %q: expected org/model or org/team/model", modelPath)
	}

	client := &http.Client{Timeout: 60 * time.Second}
	req, _ := http.NewRequestWithContext(ctx, "GET", listURL, nil)
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("NGC list: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("NGC list: HTTP %d: %s", resp.StatusCode, string(body))
	}

	var listing struct {
		ModelFiles []struct {
			Name string `json:"name"`
			Size int64  `json:"size"`
		} `json:"modelFiles"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&listing); err != nil {
		return fmt.Errorf("NGC list decode: %w", err)
	}

	dlClient := cloudDownloadClient()

	var items []downloadItem
	for _, f := range listing.ModelFiles {
		if !shouldDownloadSafetensors(f.Name) {
			continue
		}
		var dlURL string
		switch len(pathSegs) {
		case 2:
			dlURL = fmt.Sprintf("%s/models/%s/%s/versions/%s/files/%s",
				apiBase, pathSegs[0], pathSegs[1], version, f.Name)
		case 3:
			dlURL = fmt.Sprintf("%s/models/%s/%s/%s/versions/%s/files/%s",
				apiBase, pathSegs[0], pathSegs[1], pathSegs[2], version, f.Name)
		}
		items = append(items, downloadItem{
			relPath: f.Name,
			size:    f.Size,
			open: func() (io.ReadCloser, error) {
				req2, _ := http.NewRequestWithContext(ctx, "GET", dlURL, nil)
				req2.Header.Set("Authorization", "Bearer "+apiKey)
				r, err := dlClient.Do(req2)
				if err != nil {
					return nil, err
				}
				return r.Body, nil
			},
		})
	}

	return downloadItemsAsModelPack(layoutDir, "ngc", "NGC", items, storeRef, modelPath,
		fmt.Errorf("no model files found in NGC model %s", modelPath), progressKey)
}

// ---------------------------------------------------------------------------
// AWS S3  (s3://)
// ---------------------------------------------------------------------------

func pullS3(ctx context.Context, s3Ref, layoutDir, progressKey string) error {
	// s3://bucket/prefix/to/model
	withoutScheme := strings.TrimPrefix(s3Ref, "s3://")
	slashIdx := strings.Index(withoutScheme, "/")
	if slashIdx < 0 {
		return fmt.Errorf("invalid S3 ref %q: expected s3://bucket/prefix", s3Ref)
	}
	bucket := withoutScheme[:slashIdx]
	prefix := withoutScheme[slashIdx+1:]

	storeRef := s3Ref
	if reportCached(layoutDir, storeRef, withoutScheme) {
		return nil
	}

	// Build AWS config from environment (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
	// AWS_REGION, AWS_ENDPOINT_URL, AWS_S3_USE_PATH_STYLE).
	awsOpts := []func(*awsconfig.LoadOptions) error{}
	cfg, err := awsconfig.LoadDefaultConfig(ctx, awsOpts...)
	if err != nil {
		return fmt.Errorf("AWS config: %w", err)
	}

	s3opts := []func(*s3.Options){}
	if ep := os.Getenv("AWS_ENDPOINT_URL"); ep != "" {
		s3opts = append(s3opts, func(o *s3.Options) {
			o.BaseEndpoint = &ep
		})
	}
	if os.Getenv("AWS_S3_USE_PATH_STYLE") == "true" {
		s3opts = append(s3opts, func(o *s3.Options) {
			o.UsePathStyle = true
		})
	}

	client := s3.NewFromConfig(cfg, s3opts...)

	// List objects under the prefix.
	paginator := s3.NewListObjectsV2Paginator(client, &s3.ListObjectsV2Input{
		Bucket: &bucket,
		Prefix: &prefix,
	})

	type s3Object struct {
		key  string
		size int64
	}
	var objects []s3Object
	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			return fmt.Errorf("S3 list: %w", err)
		}
		for _, obj := range page.Contents {
			if obj.Key != nil && obj.Size != nil {
				objects = append(objects, s3Object{*obj.Key, *obj.Size})
			}
		}
	}

	if len(objects) == 0 {
		return fmt.Errorf("no objects found at s3://%s/%s", bucket, prefix)
	}

	var items []downloadItem
	for _, obj := range objects {
		relPath := strings.TrimPrefix(obj.key, prefix)
		relPath = strings.TrimPrefix(relPath, "/")
		if relPath == "" || !shouldDownloadSafetensors(relPath) {
			continue
		}

		key := obj.key
		items = append(items, downloadItem{
			relPath: relPath,
			size:    obj.size,
			open: func() (io.ReadCloser, error) {
				result, err := client.GetObject(ctx, &s3.GetObjectInput{
					Bucket: &bucket,
					Key:    &key,
				})
				if err != nil {
					return nil, err
				}
				return result.Body, nil
			},
		})
	}

	return downloadItemsAsModelPack(layoutDir, "s3", "S3", items, storeRef, withoutScheme,
		fmt.Errorf("no model files found at s3://%s/%s", bucket, prefix), progressKey)
}

// ---------------------------------------------------------------------------
// Google Cloud Storage  (gs://)
// ---------------------------------------------------------------------------

// pullGCS downloads a model from a GCS bucket using plain HTTP.
// Authentication: GOOGLE_APPLICATION_CREDENTIALS (service account JSON)
// or Application Default Credentials (ADC) via GOOGLE_ACCESS_TOKEN.
func pullGCS(ctx context.Context, gsRef, layoutDir, progressKey string) error {
	withoutScheme := strings.TrimPrefix(gsRef, "gs://")
	slashIdx := strings.Index(withoutScheme, "/")
	if slashIdx < 0 {
		return fmt.Errorf("invalid GCS ref %q: expected gs://bucket/prefix", gsRef)
	}
	bucket := withoutScheme[:slashIdx]
	prefix := withoutScheme[slashIdx+1:]

	storeRef := gsRef
	if reportCached(layoutDir, storeRef, withoutScheme) {
		return nil
	}

	token, err := gcsAccessToken(ctx)
	if err != nil {
		return fmt.Errorf("GCS auth: %w", err)
	}

	apiBase := "https://storage.googleapis.com/storage/v1"
	httpClient := &http.Client{Timeout: 60 * time.Second}

	// List objects.
	type gcsObject struct {
		Name string `json:"name"`
		Size string `json:"size"` // GCS returns size as string
	}
	var objects []gcsObject
	pageToken := ""
	for {
		listURL := fmt.Sprintf("%s/b/%s/o?prefix=%s&maxResults=1000", apiBase, bucket, prefix)
		if pageToken != "" {
			listURL += "&pageToken=" + pageToken
		}
		req, _ := http.NewRequestWithContext(ctx, "GET", listURL, nil)
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		resp, err := httpClient.Do(req)
		if err != nil {
			return fmt.Errorf("GCS list: %w", err)
		}
		var page struct {
			Items         []gcsObject `json:"items"`
			NextPageToken string      `json:"nextPageToken"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&page); err != nil {
			resp.Body.Close()
			return fmt.Errorf("GCS list decode: %w", err)
		}
		resp.Body.Close()
		objects = append(objects, page.Items...)
		if page.NextPageToken == "" {
			break
		}
		pageToken = page.NextPageToken
	}

	if len(objects) == 0 {
		return fmt.Errorf("no objects found at gs://%s/%s", bucket, prefix)
	}

	dlClient := cloudDownloadClient()

	var items []downloadItem
	for _, obj := range objects {
		relPath := strings.TrimPrefix(obj.Name, prefix)
		relPath = strings.TrimPrefix(relPath, "/")
		if relPath == "" || !shouldDownloadSafetensors(relPath) {
			continue
		}

		// URL-encode the object name for the download URL
		encodedName := strings.ReplaceAll(obj.Name, "/", "%2F")
		dlURL := fmt.Sprintf("%s/b/%s/o/%s?alt=media", apiBase, bucket, encodedName)

		items = append(items, downloadItem{
			relPath: relPath,
			size:    0,
			open: func() (io.ReadCloser, error) {
				req2, _ := http.NewRequestWithContext(ctx, "GET", dlURL, nil)
				if token != "" {
					req2.Header.Set("Authorization", "Bearer "+token)
				}
				r, err := dlClient.Do(req2)
				if err != nil {
					return nil, err
				}
				return r.Body, nil
			},
		})
	}

	return downloadItemsAsModelPack(layoutDir, "gs", "GCS", items, storeRef, withoutScheme,
		fmt.Errorf("no model files found at gs://%s/%s", bucket, prefix), progressKey)
}

// gcsAccessToken returns a GCS Bearer token from the environment.
// Priority: GOOGLE_ACCESS_TOKEN > GOOGLE_APPLICATION_CREDENTIALS service account.
func gcsAccessToken(ctx context.Context) (string, error) {
	// Direct token override (useful in CI or when a short-lived token is available)
	if t := os.Getenv("GOOGLE_ACCESS_TOKEN"); t != "" {
		return t, nil
	}

	// Service account JSON via GOOGLE_APPLICATION_CREDENTIALS
	saPath := os.Getenv("GOOGLE_APPLICATION_CREDENTIALS")
	if saPath == "" {
		return "", nil // anonymous / public bucket
	}

	data, err := os.ReadFile(saPath)
	if err != nil {
		return "", fmt.Errorf("read GOOGLE_APPLICATION_CREDENTIALS: %w", err)
	}

	// Extract the service account email for the token request
	var sa struct {
		ClientEmail string `json:"client_email"`
		PrivateKey  string `json:"private_key"`
		TokenURI    string `json:"token_uri"`
	}
	if err := json.Unmarshal(data, &sa); err != nil {
		return "", fmt.Errorf("parse service account JSON: %w", err)
	}

	// For a full implementation, sign a JWT and exchange for an access token.
	// As a lightweight alternative, we note that many GCP environments provide
	// the metadata server at 169.254.169.254 when ADC is available.
	req, _ := http.NewRequestWithContext(ctx, "GET",
		"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token", nil)
	req.Header.Set("Metadata-Flavor", "Google")
	resp, err := (&http.Client{Timeout: 5 * time.Second}).Do(req)
	if err == nil && resp.StatusCode == 200 {
		defer resp.Body.Close()
		var tok struct {
			AccessToken string `json:"access_token"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&tok); err == nil && tok.AccessToken != "" {
			return tok.AccessToken, nil
		}
	}

	// If metadata server not available, return empty (anonymous access).
	// Users can set GOOGLE_ACCESS_TOKEN for authenticated access.
	return "", nil
}

// ---------------------------------------------------------------------------
// Local directory  (/absolute/path)
// ---------------------------------------------------------------------------

// pullLocal imports a local model directory into the llmman OCI store as a
// CNCF ModelPack, making it available to 'llmman serve' and other commands.
func pullLocal(localPath, layoutDir string) error {
	fi, err := os.Stat(localPath)
	if err != nil {
		return fmt.Errorf("local path %q: %w", localPath, err)
	}
	if !fi.IsDir() {
		return fmt.Errorf("local path %q is not a directory", localPath)
	}

	storeRef := localPath
	if reportCached(layoutDir, storeRef, localPath) {
		return nil
	}

	var packFiles []modelPackFile
	err = filepath.Walk(localPath, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return err
		}
		rel, _ := filepath.Rel(localPath, path)
		if !shouldDownloadSafetensors(rel) {
			return nil
		}
		packFiles = append(packFiles, modelPackFile{
			localPath:    path,
			relativePath: rel,
		})
		return nil
	})
	if err != nil {
		return fmt.Errorf("walk %s: %w", localPath, err)
	}

	if len(packFiles) == 0 {
		return fmt.Errorf("no model files found in %s", localPath)
	}

	fmt.Fprintf(os.Stderr, "Importing %d files from %s\n", len(packFiles), localPath)

	// verbose=true: print a "stored <path>" line per file, since local
	// import has no download progress bar to show the work happening
	// otherwise.
	return packFilesAsModelPack(layoutDir, storeRef, filepath.Base(localPath), packFiles, true)
}
