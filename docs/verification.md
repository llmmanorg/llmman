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
full report.

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
```

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
3. Its claimed `docker-reference` is **this** repository.
4. A trusted key's signature covers the exact payload bytes.

Check 3 is not decoration. A signature is bound to a digest, not to a
location, so an attacker who wants their repository to look endorsed can
copy a legitimately signed model *and* its signature into it — the
cryptography still checks out. Comparing the claimed identity is what
pins a signature to where it was meant to live.

Repositories are compared, not full references: one digest is
legitimately reachable under many tags, and the digest binding is what
makes that safe.

## Where verification applies

| Operation | Behaviour |
|---|---|
| `pull` from an OCI registry | Policy applied. Checked *before* any layer is fetched, so a rejected model costs one manifest lookup instead of a multi-gigabyte download, then re-confirmed against the digest actually stored — a tag repointed mid-pull cannot slip past the check that just passed. Under `enforce`, a failure removes the reference again. |
| A model already in the store | Re-checked against the digest held locally, on every `pull` and on the auto-pull an inference request triggers. Being on disk is not the same as being trusted: it may have arrived before the policy existed, or under `warn`. Costs nothing when no policy applies. |
| `pull` from HuggingFace, `ms://`, `ngc://`, `s3://`, `gs://`, a local path | Not checked. There is no signature to find, and failing every such pull under `enforce` would only get the policy switched off. Use `llmman transfer --sign-key` to bring one of these into a registry as something you vouch for. |
| `transfer` from an OCI registry | Source checked under the same policy as `pull`: re-publishing an unverified model under your own name is how a supply-chain problem propagates, and a transfer that skipped the check would launder one past a policy `pull` enforces. |
| `push --sign-key` / `transfer --sign-key` | Signs the digest that was actually pushed, never one re-resolved from the destination tag afterwards. |
| `verify` | Always checks and always reports, whatever the policy says — it is the diagnostic, so its exit status reflects the signature, not the policy. |

Signing twice with different keys **appends**, so key rotation works:
publish under the new key while the old one is still trusted, then
withdraw the old key from the policy. Re-signing with a key that already
signed a digest is a no-op.

## Limitations

These are real and worth stating plainly.

- **Keyless signing (Fulcio / Rekor) is not supported.** llmman does
  key-based verification only. Keyless brings a much larger dependency
  and trust-root surface, and an air-gapped deployment — llmman's stated
  audience — cannot use it anyway. The on-registry format leaves room for
  it: a keyless signature is the same layer shape with two extra
  annotations.

- **The newer sigstore *bundle* format is not verified.** Current cosign
  publishes a bundle
  (`application/vnd.dev.sigstore.bundle.v0.3+json`) at the suffix-less
  fallback tag by default, rather than the simple-signing artifact llmman
  reads. llmman detects this and says so explicitly rather than
  misreporting the model as unsigned, and still refuses it under
  `enforce`. To produce something llmman verifies, use
  `llmman push --sign-key`.

- **`--sign-key` takes a standard PEM private key** (plain or PKCS#8
  encrypted, via `LLMMAN_SIGN_PASSWORD` / `COSIGN_PASSWORD`). cosign's own
  `cosign.key` format is not readable for *signing*. Public keys are fully
  interchangeable: a `cosign.pub` works as a trusted key here.

- **Transparency logs are not consulted**, so a signature says "this key
  signed this digest" and nothing about when, or whether the key was later
  revoked.

- **A mirrored or relocated repository cannot verify.** The claimed-identity
  check compares repositories, so a model re-hosted at
  `mirror.corp/original/name` will not match a signature naming
  `original/name`. That is the check working as intended — it is what stops
  a signature being copied into someone else's repository — but it means a
  pull-through mirror needs its own signatures today. A per-rule identity
  override would close this and is not implemented.

- **Signing happens in the CLI, never in the daemon.** `llmman push
  --sign-key` has the daemon report which digest it pushed and then signs
  that itself, so no key path or passphrase ever crosses the socket. The
  daemon binds unauthenticated TCP loopback, which is not user-scoped and
  carries no peer credentials — accepting a key path there would let any
  local user have it read a file only its own user can read.
