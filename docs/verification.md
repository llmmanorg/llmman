# Signature verification

llmman verifies the *integrity* of everything it pulls: every blob is
content-addressed and re-hashed on the way in, so the bytes always match
what the registry said they would be. That proves nothing about **who
put them there**.

That gap matters more for llmman than for a registry with a curated
library, because llmman deliberately has no gatekeeper. A bare
`llmman pull gemma4` resolves through the `[aliases]` table of
`llmman.conf` — which either config tier can repoint — to some registry
you never typed, and the GGUF it lands on goes straight into
`llama-server`'s parser.

Signatures close that gap, and they do it per-artifact rather than
per-registry: the check works the same on Docker Hub, GHCR, quay, or your
own air-gapped mirror.

## Format

llmman reads and writes cosign's **simple signing** format. For a
manifest with digest `sha256:<hex>` in repository `<repo>`, the signature
lives at the tag `<repo>:sha256-<hex>.sig` as an ordinary OCI image
manifest, with one layer per signature:

| | |
|---|---|
| layer media type | `application/vnd.dev.cosign.simplesigning.v1+json` |
| layer blob | the signed payload naming the manifest digest and repository |
| layer annotation | `dev.cosignproject.cosign/signature` — base64 of the raw signature |
| keyless only | `dev.sigstore.cosign/certificate`, `dev.sigstore.cosign/chain`, `dev.sigstore.cosign/bundle`: the Fulcio certificate, its chain, and the Rekor bundle, all PEM or JSON in annotations |

Nothing about this is llmman-specific: `cosign verify --key` reads what
`llmman push --sign-key` writes. There is a test for exactly that claim
(`go-shim/sigstore_cosign_test.go`), run against the real `cosign` binary
rather than against our own reading of its documentation.

## Quick start

Generate a key pair. Any standard PEM works — this is plain PKCS#8, which
is what `openssl` produces:

```console
$ openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 -out signing.key
$ openssl pkey -in signing.key -pubout -out signing.pub
```

Sign on the way out:

```console
$ llmman push docker.io/myorg/mymodel:v1 --sign-key signing.key
$ llmman transfer hf.co/owner/model docker.io/myorg/mymodel:v1 --sign-key signing.key
```

Check by hand at any time — no daemon, no local copy required:

```console
$ llmman verify docker.io/myorg/mymodel:v1 --key signing.pub
Reference:  docker.io/myorg/mymodel:v1
Digest:     sha256:9f2c...
Signatures: 1
Verified:   yes
  signed by signing.pub claiming docker.io/myorg/mymodel
```

`llmman verify` exits non-zero when a model is unsigned or signed by an
untrusted key, so it drops straight into a pipeline. `--json` prints the
full report. With `--key` only the given keys are checked: the keys and
the keyless identity a matching `[[verify.trust]]` rule names are set
aside for that run, and only the rule's `claims` still applies.

## Enforcing it on pull

Verification during `pull` is driven by the `[verify]` section of
`llmman.conf`, read from `/etc/llmman/` then `~/.config/llmman/` (later
files win — see [configuration.md](configuration.md#llmmanconf)):

```toml
[verify]
# Applies to any reference no rule below matches. Default "off".
default = "off"

[[verify.trust]]
pattern = "docker.io/myorg/**"
keys    = ["keys/myorg.pub"]     # relative paths resolve against this file
mode    = "enforce"

[[verify.trust]]
# Narrower rules written later override broader earlier ones, and
# *replace* their key set rather than adding to it — so "trust the org
# key everywhere except here" is expressible.
pattern = "docker.io/myorg/experimental"
keys    = ["keys/lab.pub"]
mode    = "warn"

[[verify.trust]]
pattern = "docker.io/ai/*"
keys    = ["~/.config/llmman/keys/docker-ai.pub"]
mode    = "warn"

[[verify.trust]]
# A relocated copy of docker.io/myorg/**: the pulled reference selects the
# rule, `claims` says which repository the signature may name.
pattern = "mirror.corp/**"
claims  = "docker.io/myorg/**"
keys    = ["keys/myorg.pub"]
mode    = "enforce"
```

### Keyless signatures

A rule may name a signing identity instead of keys. The signature is then
one a Fulcio certificate made (`cosign sign` without `--key`), and the
layer carries that certificate, its chain and a Rekor bundle in the three
`dev.sigstore.cosign/*` annotations above. cosign v3 publishes a Sigstore
bundle by default, and this layout needs two flags, both marked deprecated
there (reading the bundle format is a second PR):

```sh
cosign sign --new-bundle-format=false --use-signing-config=false <ref>
```

The roots the certificate and the bundle are checked against come from a
Sigstore TrustedRoot file named once under `[verify]`. For the public-good
instance, `cosign trusted-root create --with-default-services --out
trusted_root.json` writes the current file from the Sigstore TUF
repository, and the same JSON is distributed as `trusted_root.json` in
[sigstore/root-signing](https://github.com/sigstore/root-signing).
The file keeps each earlier Fulcio CA and Rekor key with its validity
window, and llmman uses the ones valid at the signing time, so a signature
made while an earlier key was current still verifies as long as the file
still lists that key.

```toml
[verify]
default      = "off"
trusted_root = "sigstore/trusted_root.json"   # relative to this file

[[verify.trust]]
pattern = "docker.io/myorg/**"
issuer  = "https://token.actions.githubusercontent.com"
subject = "https://github.com/myorg/models/.github/workflows/sign.yml@refs/heads/main"
mode    = "enforce"
```

`issuer` and `subject` are compared exactly and both are required: an
issuer alone would accept every identity that issuer signs for, and a
subject alone the same name from any issuer. A rule sets `keys` or an
identity, not both. An identity rule without a `trusted_root` is a broken
configuration, never an unsigned model: under `enforce` the pull is
refused, under `warn` it proceeds with a warning naming `trusted_root`,
and in neither mode is the model reported as "not signed".

Everything is checked offline, in the order the Sigstore client
specification gives:

1. The bundle's signed entry timestamp verifies under the log key the
   trusted root holds for the bundle's `logID`, with the hash that log's
   `keyDetails` declares. Only then is its `integratedTime` believed as
   the moment of signing. A trusted root with no `tlogs`, like one with
   no `certificateAuthorities`, is a configuration error.
2. The certificate chains to a Fulcio root from the trusted root, valid at
   that moment, with the `digitalSignature` key usage and the code-signing
   extended key usage, both required explicitly. The chain annotation
   supplies intermediates only; a self-signed certificate the registry
   serves is never a root.
3. The certificate's OIDC issuer extension equals `issuer` and one of its
   subject alternative names equals `subject`: an email, a URI, a DNS or
   IP name, or the otherName (`1.3.6.1.4.1.57264.1.7`) Fulcio writes a
   username identity as.
4. The logged `hashedrekord` entry names this layer's signature, this
   certificate and the SHA-256 of this payload, so the timestamp vouches
   for this signature and not another one by the same signer.
5. The signature verifies under the certificate's key over the payload
   bytes, and the payload passes the same digest and identity checks a
   key-based signature does.

The certificate's SCT is not checked (the trusted root's `ctlogs` are not
consulted), the `.sig` layout carries no inclusion proof, and nothing is
fetched from a transparency log. The trusted root file is what you trust in its place;
keep it current the way you would a CA bundle.

| Mode | Effect on `pull` |
|---|---|
| `off` | No check at all. |
| `warn` | Check, print the outcome, pull anyway. |
| `enforce` | Refuse to pull unless a trusted key signed that exact manifest. |

A `[[verify.trust]]` rule with no explicit `mode` defaults to `warn`.

**Patterns** match the *repository* (`host/namespace/name`, no tag), since
trust is about who publishes a model and tags move. `*` matches within one
path segment; `**` crosses them. So `docker.io/myorg/*` covers
`docker.io/myorg/model` but not `docker.io/myorg/team/model`, and
`docker.io/myorg/**` covers both.

The optional `claims` on a rule is a glob with the same grammar, matched
against the repository the signature itself names rather than the pulled
reference. In the last rule above, pulling `mirror.corp/myorg/name` selects
the rule by `pattern`, and a signature naming `docker.io/myorg/name` is
accepted because that identity matches `claims`. With `claims` unset the
signed repository must equal the pulled one exactly.

`LLMMAN_VERIFY=off|warn|enforce` overrides the mode — but not the keys —
for every reference, which is the useful knob in CI: a pipeline can demand
`enforce` without editing the image's config files.

### Why the default is `off`

A default of `warn` sounds safer and is worse. With no configured trust
roots there is nothing any model could be checked against, so every pull
would print an unsigned-model warning you can do nothing about — a
reliable way to teach people to ignore the one warning that eventually
matters. A trust policy only means anything once you have said whom you
trust, so that is what turns it on.

## What is actually checked

For each published signature, all four of these must hold:

1. The payload is a cosign simple-signing document (`critical.type`).
2. It names **this** manifest digest, not some other one in the repository.
3. Its claimed `docker-reference` is **this** repository, by default. Under
   a rule with `claims`, the claimed repository must match the `claims`
   glob instead, while the pulled reference must match `pattern`.
4. A trusted key's signature covers the exact payload bytes. Under an
   identity rule the key is the one in the `dev.sigstore.cosign/certificate`
   annotation, accepted only after the `chain` and `bundle` annotations
   pass the keyless checks above.

Check 3 is not decoration. A signature is bound to a digest, not to a
location, so an attacker who wants their repository to look endorsed can
copy a legitimately signed model *and* its signature into it — the
cryptography still checks out. Comparing the claimed identity to the
pulled repository is what pins a signature to where it was meant to live.

`claims` loosens that pin on purpose. The signed repository is then not
related to the pulled one: any repository matching `claims` may
authenticate any reference matching `pattern`. Scope `pattern` to the
repositories the relocated copies actually live in, and `claims` to the
repositories whose signatures they may carry, rather than to a whole
host.

Repositories are compared, not full references: one digest is
legitimately reachable under many tags, and the digest binding is what
makes that safe.

## Where verification applies

| Operation | Behaviour |
|---|---|
| `pull` from an OCI registry | Policy applied. Checked *before* any layer is fetched, so a rejected model costs one manifest lookup instead of a multi-gigabyte download, then re-confirmed against the digest actually stored — a tag repointed mid-pull cannot slip past the check that just passed. Under `enforce`, a failure removes the reference again. |
| A model already in the store | Re-checked against the digest held locally, on every `pull` and on the auto-pull an inference request triggers. Being on disk is not the same as being trusted: it may have arrived before the policy existed, or under `warn`. Costs nothing when no policy applies. |
| `pull` from HuggingFace, `ms://`, `ngc://`, `s3://`, `gs://`, a local path | Not checked. There is no signature to find, and failing every such pull under `enforce` would only get the policy switched off. Use `llmman transfer --sign-key` to bring one of these into a registry as something you vouch for. |
| `resolve` (the vLLM plugin, `llmman-csi`) | The same as `pull`, to a model it pulls and to one already in the store. |
| `transfer` from an OCI registry | Source checked under the same policy as `pull`: re-publishing an unverified model under your own name is how a supply-chain problem propagates, and a transfer that skipped the check would launder one past a policy `pull` enforces. |
| `push --sign-key` / `transfer --sign-key` | Signs the digest that was actually pushed, never one re-resolved from the destination tag afterwards. |
| A cold load under `[verify] layers` | The stored manifest, its config blob and every stored layer blob are re-hashed and compared with the digests the manifest records. `warn` logs a notice per mismatch; `enforce` fails the load. Off by default; the signature policy never turns it on. See below. |
| `verify` | Always checks and always reports, whatever the policy says — it is the diagnostic, so its exit status reflects the signature, not the policy. |

Signing twice with different keys **appends**, so key rotation works:
publish under the new key while the old one is still trusted, then
withdraw the old key from the policy. Re-signing with a key that already
signed a digest is a no-op.

## Layer verification at load

Signature verification binds a manifest digest to a key. It says nothing
about whether the bytes under `blobs/sha256/` still match that manifest:
a layer's SHA-256 is checked once, while it is downloaded, and from then
on the file is trusted by name. `[verify] layers = "warn"` or `"enforce"`
(or `LLMMAN_VERIFY_LAYERS`) adds a second check on every cold load of
`llmman serve` and on `llmman resolve`: the stored manifest, its config
blob and each stored layer blob are hashed again and compared with the
digest the manifest records. A blob that is
missing, cannot be read, or carries a malformed digest counts as a
mismatch, with the cause in the notice; a manifest whose own bytes
mismatch is reported alone, since the layer list it carries is not
trusted. The manifest is resolved once per load: the running-model
check, the layer check and the load all use that one descriptor, and
the load opens it pinned by digest, so a tag repointed in between is
not found rather than loaded unchecked; a model stored under an
absolute path or a URI cannot carry a digest and is loaded by name, so
that guarantee does not extend to it. Each blob is opened without
following symlinks and must be a regular file, checked on the open
handle, so a blob replaced by a symlink, a directory, a FIFO or a
device is a mismatch rather than something followed or read.

Under `warn` each mismatch logs a notice naming the reference, the digest
and the media type, and the load proceeds. Under `enforce` the load fails
with an error listing every mismatched digest. Either way nothing is
deleted and no registry is contacted: the blob may be the operator's only
copy, and the operator asked to be told, not to have it removed. This is
deliberately unlike a failed signature check under `enforce`, which
removes the reference.

The check is independent of `default` and the trust rules, needs no keys,
and applies to every model in the store whatever its origin, including
ones pulled from HuggingFace or a local path. It runs once per cold load,
not per request. The cost is one sequential read and hash of every
blob, once per cold load.

## Limitations

These are real and worth stating plainly.

- **Layers are checked in the store, not where they are served from.**
  The check hashes the blobs under `blobs/sha256/`. Several layer kinds
  are served from a copy under the cache directory instead: a GGUF that
  arrived inside a tar, a projector, and any layer with a file path
  annotation. That copy is a hardlink when the cache shares a filesystem
  with the store and a real copy otherwise, and an existing copy is
  reused on a size match alone. The manifest records no digest for it,
  so a cache copy is not re-hashed.

- **A write after the check is not detected.** Layer verification hashes
  a blob before the backend opens it; a change made after that point, or
  while the model is being served, goes unnoticed until the next cold
  load. Kernel-enforced read-time integrity (fs-verity) could close that
  gap later.

- **Keyless verification is offline and partial.** The certificate's
  SCT is not checked, no inclusion proof is checked, and no transparency
  log is consulted online. What stands in for them is the signed entry
  timestamp verified against the log key in `trusted_root`, so a stale or
  wrong trusted root file is the weak point. There is no keyless
  *signing*: `llmman push --sign-key` is key-based, and the keyless
  signatures llmman verifies are the ones `cosign sign` publishes with
  the two legacy flags shown above.

- **The newer sigstore *bundle* format is not verified.** Current cosign
  publishes a bundle
  (`application/vnd.dev.sigstore.bundle.v0.3+json`) at the suffix-less
  fallback tag by default, rather than the simple-signing artifact llmman
  reads. llmman detects this and says so explicitly rather than
  misreporting the model as unsigned, and still refuses it under
  `enforce`. To produce something llmman verifies, use
  `llmman push --sign-key`, or `cosign sign` with the two legacy flags.

- **`--sign-key` takes a standard PEM private key** (plain or PKCS#8
  encrypted, via `LLMMAN_SIGN_PASSWORD` / `COSIGN_PASSWORD`). cosign's own
  `cosign.key` format is not readable for *signing*. Public keys are fully
  interchangeable: a `cosign.pub` works as a trusted key here.

- **Transparency logs are not consulted online.** A key-based signature
  says "this key signed this digest" and nothing about when, or whether
  the key was later revoked. A keyless signature carries the log's signed
  entry timestamp, which is verified offline against the log key in
  `trusted_root` and fixes the moment of signing; whether the entry is
  actually in the log is not checked.

- **A relocated repository verifies only through `claims`.** The
  claimed-identity check compares repositories, so a model re-hosted at
  `mirror.corp/myorg/name` will not match a signature naming
  `docker.io/myorg/name` unless the matching `[[verify.trust]]` rule sets
  `claims` to a glob the signed repository satisfies. The claimed repository
  is canonicalised before matching, so `docker.io/myorg/**` also covers the
  `index.docker.io/myorg/name` identity cosign writes. `pattern` still
  selects the rule by the pulled reference, and `claims` unset keeps exact
  equality. The signature artifact is looked up under the pulled
  repository's `sha256-<hex>.sig` tag, so a relocated copy verifies only if
  that tag was copied alongside the manifest. When the reference does not
  change, because a pull of `docker.io/...` is merely served by another
  host, registry mirrors (`[registries."<host>"]` `mirrors`) are the
  mechanism and `claims` is not needed.

- **Signing happens in the CLI, never in the daemon.** `llmman push
  --sign-key` has the daemon report which digest it pushed and then signs
  that itself, so no key path or passphrase ever crosses the socket. The
  daemon binds unauthenticated TCP loopback, which is not user-scoped and
  carries no peer credentials — accepting a key path there would let any
  local user have it read a file only its own user can read.
