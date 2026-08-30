//! AWS S3 and S3-compatible object stores (`s3://bucket/prefix`).
//!
//! The official SDK rather than a hand-rolled signer, so the whole
//! credential chain behaves as it did under aws-sdk-go-v2: env vars,
//! `~/.aws` profiles, SSO, `credential_process`, assume-role and
//! EC2/ECS instance roles. `AWS_ENDPOINT_URL` and
//! `AWS_S3_USE_PATH_STYLE=true` point it at MinIO and friends.

use anyhow::{Context, Result};
use aws_config::BehaviorVersion;

use super::Target;

/// `reference` is the full `s3://…` reference (kept verbatim as the
/// stored ref); `rest` is it with the scheme stripped.
pub(crate) async fn pull(reference: &str, rest: &str, target: &Target<'_>) -> Result<()> {
    let (bucket, prefix) = super::split_bucket_prefix(rest, reference)?;

    if target.report_cached(reference, rest) {
        return Ok(());
    }

    // Pinned explicitly rather than via the `behavior-version-latest`
    // feature, so an SDK upgrade changing a default is a visible edit to
    // this line, not a silent change on `cargo update`.
    let config = aws_config::defaults(BehaviorVersion::v2026_01_12())
        .load()
        .await;
    let mut s3 = aws_sdk_s3::config::Builder::from(&config);
    if let Ok(endpoint) = std::env::var("AWS_ENDPOINT_URL") {
        if !endpoint.is_empty() {
            s3 = s3.endpoint_url(endpoint);
        }
    }
    if std::env::var("AWS_S3_USE_PATH_STYLE").as_deref() == Ok("true") {
        s3 = s3.force_path_style(true);
    }
    let client = aws_sdk_s3::Client::from_conf(s3.build());

    let mut pages = client
        .list_objects_v2()
        .bucket(bucket)
        .prefix(prefix)
        .into_paginator()
        .send();
    let mut keys: Vec<(String, i64)> = Vec::new();
    while let Some(page) = pages.next().await {
        let page = page.map_err(anyhow::Error::from).context("S3 list")?;
        for obj in page.contents() {
            if let (Some(key), Some(size)) = (obj.key(), obj.size()) {
                keys.push((key.to_string(), size));
            }
        }
    }
    if keys.is_empty() {
        anyhow::bail!("no objects found at s3://{bucket}/{prefix}");
    }

    let mut packed = Vec::new();
    for (key, size) in &keys {
        let Some(rel_path) = super::relative_to_prefix(key, prefix) else {
            continue;
        };
        if !super::should_pack(rel_path) {
            continue;
        }
        let object = client
            .get_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map_err(anyhow::Error::from)
            .with_context(|| format!("S3 download {key}"))?;
        // ByteStream → AsyncRead → Bytes stream, so this shares the one
        // staging/progress loop every other source uses.
        let body = tokio_util::io::ReaderStream::new(object.body.into_async_read());
        packed.push(super::download_to_pack_file(target, "s3", "S3", rel_path, *size, body).await?);
    }

    super::pack_as_model_pack(
        target,
        reference,
        rest,
        packed,
        format!("no model files found at s3://{bucket}/{prefix}"),
    )
}
